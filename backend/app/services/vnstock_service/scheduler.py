from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import delete, select

from app.db.database import async_session
from app.db.models import ScheduledSyncJob, ScheduledSyncJobRun
from app.services.sync_status import HistoryJobStatusData, sync_status

from .core import logger

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
UTC = timezone.utc


class ScheduledSyncType(str, Enum):
    HISTORY = "history"
    FINANCE = "finance"
    COMPANY = "company"


class ScheduledSyncAction(str, Enum):
    SYNC = "sync"
    AUDIT = "audit"
    REPAIR = "repair"
    FULL = "full"
    QUICK = "quick"


class ScheduledSyncIntervalUnit(str, Enum):
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"


class ScheduledSyncRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL_SUCCEEDED = "partial_succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class SchedulerDispatchResult:
    status: ScheduledSyncRunStatus
    summary: Dict[str, Any]
    error: str | None = None


class SchedulerValidationError(ValueError):
    """Raised when a scheduler payload is invalid."""


class SchedulerNotFoundError(SchedulerValidationError):
    """Raised when a scheduler row cannot be found."""


class DeferredRunError(RuntimeError):
    """Raised when a queued run should remain queued and be retried soon."""

    def __init__(self, delay_seconds: float = 10.0):
        super().__init__("Scheduled run deferred")
        self.delay_seconds = max(1.0, float(delay_seconds))


def utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)


def to_utc_naive(local_dt: datetime) -> datetime:
    if local_dt.tzinfo is None:
        aware = local_dt.replace(tzinfo=VN_TZ)
    else:
        aware = local_dt.astimezone(VN_TZ)
    return aware.astimezone(UTC).replace(tzinfo=None)


def utc_naive_to_local_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value.replace(tzinfo=UTC).astimezone(VN_TZ)
    return aware.isoformat(timespec="seconds")


class ScheduledSyncService:
    """Persistent in-process scheduler for existing sync operations."""

    DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"

    def __init__(
        self,
        *,
        run_history_sync: Callable[..., Awaitable[Dict[str, Any]]],
        run_history_audit_sync: Callable[..., Awaitable[Dict[str, Any]]],
        run_history_repair_sync: Callable[..., Awaitable[Dict[str, Any]]],
        run_finance_sync: Callable[..., Awaitable[Dict[str, Any]]],
        run_company_sync: Callable[..., Awaitable[Dict[str, Any]]],
        poll_interval_seconds: float = 5.0,
        status_poll_seconds: float = 2.0,
        deferred_poll_seconds: float = 10.0,
        retry_base_minutes: int = 15,
    ) -> None:
        self._run_history_sync = run_history_sync
        self._run_history_audit_sync = run_history_audit_sync
        self._run_history_repair_sync = run_history_repair_sync
        self._run_finance_sync = run_finance_sync
        self._run_company_sync = run_company_sync
        self._poll_interval_seconds = max(1.0, float(poll_interval_seconds))
        self._status_poll_seconds = max(0.5, float(status_poll_seconds))
        self._deferred_poll_seconds = max(1.0, float(deferred_poll_seconds))
        self._retry_base_minutes = max(1, int(retry_base_minutes))
        self._loop_task: asyncio.Task | None = None
        self._current_run_task: asyncio.Task | None = None
        self._startup_lock = asyncio.Lock()

    async def start_background_tasks(self) -> None:
        async with self._startup_lock:
            if self._loop_task and not self._loop_task.done():
                return
            await self.recover_incomplete_runs()
            self._loop_task = asyncio.create_task(self._scheduler_loop())

    async def stop_background_tasks(self) -> None:
        if self._current_run_task and not self._current_run_task.done():
            self._current_run_task.cancel()
            await asyncio.gather(self._current_run_task, return_exceptions=True)
        self._current_run_task = None

        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        self._loop_task = None

    async def recover_incomplete_runs(self) -> None:
        """Mark stale running attempts as failed after process restarts."""
        now = utc_now()
        async with async_session() as session:
            stmt = select(ScheduledSyncJobRun).where(
                ScheduledSyncJobRun.status == ScheduledSyncRunStatus.RUNNING.value
            )
            rows = (await session.execute(stmt)).scalars().all()
            for row in rows:
                row.status = ScheduledSyncRunStatus.FAILED.value
                row.finished_at = now
                row.error = "Scheduler process restarted while this run was in progress"
                row.summary = {"message": "Recovered as failed after scheduler restart"}
            await session.commit()

    async def list_jobs(self) -> list[dict[str, Any]]:
        async with async_session() as session:
            stmt = select(ScheduledSyncJob).order_by(
                ScheduledSyncJob.next_run_at.asc(),
                ScheduledSyncJob.id.asc(),
            )
            jobs = (await session.execute(stmt)).scalars().all()
            return [self._serialize_job(job) for job in jobs]

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = min(200, max(1, int(limit)))
        async with async_session() as session:
            stmt = (
                select(ScheduledSyncJobRun, ScheduledSyncJob)
                .join(ScheduledSyncJob, ScheduledSyncJob.id == ScheduledSyncJobRun.job_id)
                .order_by(
                    ScheduledSyncJobRun.scheduled_for.desc(),
                    ScheduledSyncJobRun.id.desc(),
                )
                .limit(safe_limit)
            )
            rows = (await session.execute(stmt)).all()
            return [
                self._serialize_run(run, job)
                for run, job in rows
            ]

    async def create_job(self, payload: Dict[str, Any]) -> dict[str, Any]:
        normalized = self._validate_payload(payload)
        async with async_session() as session:
            job = ScheduledSyncJob(**normalized)
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return self._serialize_job(job)

    async def update_job(self, job_id: int, payload: Dict[str, Any]) -> dict[str, Any]:
        async with async_session() as session:
            job = await session.get(ScheduledSyncJob, job_id)
            if job is None:
                raise SchedulerNotFoundError("Scheduled job not found")

            merged_payload = self._merge_job_payload(job, payload)
            normalized = self._validate_payload(merged_payload)
            for key, value in normalized.items():
                setattr(job, key, value)

            # Drop queued attempts because the job definition just changed.
            await session.execute(
                delete(ScheduledSyncJobRun).where(
                    ScheduledSyncJobRun.job_id == job_id,
                    ScheduledSyncJobRun.status == ScheduledSyncRunStatus.QUEUED.value,
                )
            )
            await session.commit()
            await session.refresh(job)
            return self._serialize_job(job)

    async def delete_job(self, job_id: int) -> None:
        async with async_session() as session:
            job = await session.get(ScheduledSyncJob, job_id)
            if job is None:
                raise SchedulerNotFoundError("Scheduled job not found")
            await session.delete(job)
            await session.commit()

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._enqueue_due_runs()
                await self._cleanup_current_run_task()
                await self._maybe_start_next_run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Scheduled sync loop error: %s", exc)
            await asyncio.sleep(self._poll_interval_seconds)

    async def _cleanup_current_run_task(self) -> None:
        if self._current_run_task and self._current_run_task.done():
            await asyncio.gather(self._current_run_task, return_exceptions=True)
            self._current_run_task = None

    async def _enqueue_due_runs(self) -> None:
        now = utc_now()
        async with async_session() as session:
            stmt = (
                select(ScheduledSyncJob)
                .where(
                    ScheduledSyncJob.enabled.is_(True),
                    ScheduledSyncJob.next_run_at <= now,
                )
                .order_by(
                    ScheduledSyncJob.next_run_at.asc(),
                    ScheduledSyncJob.id.asc(),
                )
            )
            jobs = (await session.execute(stmt)).scalars().all()
            for job in jobs:
                while job.next_run_at <= now:
                    queued_run = ScheduledSyncJobRun(
                        job_id=job.id,
                        attempt_number=1,
                        status=ScheduledSyncRunStatus.QUEUED.value,
                        scheduled_for=job.next_run_at,
                        summary={},
                    )
                    session.add(queued_run)
                    job.next_run_at = self._advance_utc_occurrence(
                        current_utc=job.next_run_at,
                        interval_value=job.interval_value,
                        interval_unit=job.interval_unit,
                    )
            await session.commit()

    async def _maybe_start_next_run(self) -> None:
        if self._current_run_task and not self._current_run_task.done():
            return
        if self._is_any_sync_running():
            return

        now = utc_now()
        async with async_session() as session:
            stmt = (
                select(ScheduledSyncJobRun.id)
                .join(ScheduledSyncJob, ScheduledSyncJob.id == ScheduledSyncJobRun.job_id)
                .where(
                    ScheduledSyncJob.enabled.is_(True),
                    ScheduledSyncJobRun.status == ScheduledSyncRunStatus.QUEUED.value,
                    ScheduledSyncJobRun.scheduled_for <= now,
                )
                .order_by(
                    ScheduledSyncJobRun.scheduled_for.asc(),
                    ScheduledSyncJobRun.id.asc(),
                )
                .limit(1)
            )
            run_id = (await session.execute(stmt)).scalar_one_or_none()

        if run_id is not None:
            self._current_run_task = asyncio.create_task(self._run_queued_job(run_id))

    async def _run_queued_job(self, run_id: int) -> None:
        async with async_session() as session:
            run = await session.get(ScheduledSyncJobRun, run_id)
            if run is None or run.status != ScheduledSyncRunStatus.QUEUED.value:
                return
            job = await session.get(ScheduledSyncJob, run.job_id)
            if job is None or not job.enabled:
                return
            run.status = ScheduledSyncRunStatus.RUNNING.value
            run.started_at = utc_now()
            run.error = None
            run.summary = {}
            await session.commit()

        try:
            result = await self._execute_job(job)
        except DeferredRunError as exc:
            await self._defer_run(run_id, exc.delay_seconds)
            return
        except asyncio.CancelledError:
            await self._mark_run_failed(
                run_id=run_id,
                job_id=job.id,
                error="Scheduled run cancelled",
                summary={"message": "Scheduled run cancelled"},
                schedule_retry=True,
            )
            raise
        except Exception as exc:
            await self._mark_run_failed(
                run_id=run_id,
                job_id=job.id,
                error=str(exc)[:1000],
                summary={"message": str(exc)[:1000]},
                schedule_retry=True,
            )
            return

        if result.status == ScheduledSyncRunStatus.FAILED:
            await self._mark_run_failed(
                run_id=run_id,
                job_id=job.id,
                error=result.error or "Scheduled sync failed",
                summary=result.summary,
                schedule_retry=True,
            )
            return

        await self._mark_run_completed(
            run_id=run_id,
            job_id=job.id,
            status=result.status,
            summary=result.summary,
        )

    async def _defer_run(self, run_id: int, delay_seconds: float) -> None:
        async with async_session() as session:
            run = await session.get(ScheduledSyncJobRun, run_id)
            if run is None:
                return
            run.status = ScheduledSyncRunStatus.QUEUED.value
            run.started_at = None
            run.finished_at = None
            run.scheduled_for = utc_now() + timedelta(seconds=delay_seconds)
            run.error = None
            run.summary = {}
            await session.commit()

    async def _mark_run_completed(
        self,
        run_id: int,
        job_id: int,
        status: ScheduledSyncRunStatus,
        summary: Dict[str, Any],
    ) -> None:
        finished_at = utc_now()
        async with async_session() as session:
            run = await session.get(ScheduledSyncJobRun, run_id)
            job = await session.get(ScheduledSyncJob, job_id)
            if run is None or job is None:
                return
            run.status = status.value
            run.finished_at = finished_at
            run.error = None
            run.summary = summary
            job.last_run_at = finished_at
            await session.commit()

    async def _mark_run_failed(
        self,
        *,
        run_id: int,
        job_id: int,
        error: str,
        summary: Dict[str, Any],
        schedule_retry: bool,
    ) -> None:
        finished_at = utc_now()
        async with async_session() as session:
            run = await session.get(ScheduledSyncJobRun, run_id)
            job = await session.get(ScheduledSyncJob, job_id)
            if run is None or job is None:
                return

            run.status = ScheduledSyncRunStatus.FAILED.value
            run.finished_at = finished_at
            run.error = error
            run.summary = summary
            job.last_run_at = finished_at

            if schedule_retry and run.attempt_number <= job.max_retries:
                retry_delay = timedelta(minutes=self._retry_base_minutes * run.attempt_number)
                session.add(
                    ScheduledSyncJobRun(
                        job_id=job.id,
                        attempt_number=run.attempt_number + 1,
                        status=ScheduledSyncRunStatus.QUEUED.value,
                        scheduled_for=utc_now() + retry_delay,
                        summary={},
                    )
                )
            await session.commit()

    async def _execute_job(self, job: ScheduledSyncJob) -> SchedulerDispatchResult:
        if self._is_any_sync_running():
            raise DeferredRunError(delay_seconds=self._deferred_poll_seconds)

        symbols = list(job.symbols or [])
        index_symbol = job.index_symbol or None

        if job.sync_type == ScheduledSyncType.HISTORY.value:
            if job.sync_action == ScheduledSyncAction.SYNC.value:
                response = await self._run_history_sync(
                    force_restart=False,
                    symbols=symbols or None,
                    index_symbol=index_symbol,
                )
                await self._ensure_started(response)
                snapshot = await self._monitor_symbol_job("history_sync")
                summary = self._build_runtime_summary(snapshot, "history")
                return self._classify_completed_run(job=job, summary=summary, label="History sync")
            if job.sync_action == ScheduledSyncAction.AUDIT.value:
                response = await self._run_history_audit_sync(
                    symbols=symbols or None,
                    start_date=job.date_from.isoformat(),
                    end_date=job.date_to.isoformat(),
                    auto_repair=job.auto_repair,
                    index_symbol=index_symbol,
                )
                await self._ensure_started(response, "History audit")
                return self._classify_completed_run(job=job, summary=response, label="History audit")
            if job.sync_action == ScheduledSyncAction.REPAIR.value:
                response = await self._run_history_repair_sync(
                    symbols=symbols or None,
                    start_date=job.date_from.isoformat(),
                    end_date=job.date_to.isoformat(),
                    index_symbol=index_symbol,
                )
                await self._ensure_started(response, "History repair")
                return self._classify_completed_run(job=job, summary=response, label="History repair")

        if job.sync_type == ScheduledSyncType.FINANCE.value:
            response = await self._run_finance_sync(
                force_restart=False,
                symbols=symbols or None,
                index_symbol=index_symbol,
                quick_sync=job.sync_action == ScheduledSyncAction.QUICK.value,
            )
            await self._ensure_started(response)
            snapshot = await self._monitor_symbol_job("finance_sync")
            summary = self._build_runtime_summary(snapshot, "finance")
            return self._classify_completed_run(job=job, summary=summary, label="Finance sync")

        if job.sync_type == ScheduledSyncType.COMPANY.value:
            response = await self._run_company_sync(
                force_restart=False,
                symbols=symbols or None,
                index_symbol=index_symbol,
                quick_sync=job.sync_action == ScheduledSyncAction.QUICK.value,
            )
            await self._ensure_started(response)
            snapshot = await self._monitor_symbol_job("company_sync")
            summary = self._build_runtime_summary(snapshot, "company")
            return self._classify_completed_run(job=job, summary=summary, label="Company sync")

        raise SchedulerValidationError("Unsupported scheduled sync type/action")

    async def _ensure_started(self, response: Dict[str, Any], label: str = "Scheduled sync") -> None:
        if response.get("started"):
            return
        if response.get("state") == "running":
            raise DeferredRunError(delay_seconds=self._deferred_poll_seconds)
        raise RuntimeError(response.get("message") or f"{label} did not start")

    async def _monitor_symbol_job(self, runtime_name: str) -> HistoryJobStatusData:
        seen_running = False
        started_wait_started_at = datetime.now(tz=UTC)
        while True:
            runtime = getattr(sync_status, runtime_name)
            if runtime.is_running:
                seen_running = True
            elif seen_running:
                if runtime.error and runtime.processed_symbols < runtime.total_symbols:
                    raise RuntimeError(runtime.error)
                return runtime
            else:
                elapsed = datetime.now(tz=UTC) - started_wait_started_at
                if elapsed.total_seconds() > 30:
                    if runtime.error:
                        raise RuntimeError(runtime.error)
                    return runtime
            await asyncio.sleep(self._status_poll_seconds)

    def _build_runtime_summary(self, runtime: HistoryJobStatusData, label: str) -> Dict[str, Any]:
        return {
            "type": label,
            "total_symbols": runtime.total_symbols,
            "processed_symbols": runtime.processed_symbols,
            "success_symbols": runtime.success_symbols,
            "failed_symbols": runtime.failed_symbols,
            "failed_tickers": list(runtime.failed_tickers),
            "current_symbol": runtime.current_symbol,
            "started_at": runtime.started_at,
            "last_run_at": runtime.last_run_at,
            "error": runtime.error,
            "progress": runtime.progress,
        }

    def _is_any_sync_running(self) -> bool:
        return any([
            sync_status.history_sync.is_running,
            sync_status.history_audit.is_running,
            sync_status.history_repair.is_running,
            sync_status.finance_sync.is_running,
            sync_status.company_sync.is_running,
        ])

    def _validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise SchedulerValidationError("Job name is required")

        sync_type = str(payload.get("sync_type") or "").strip().lower()
        sync_action = str(payload.get("sync_action") or "").strip().lower()
        try:
            sync_type_enum = ScheduledSyncType(sync_type)
        except ValueError as exc:
            raise SchedulerValidationError("Invalid sync_type") from exc

        valid_actions = {
            ScheduledSyncType.HISTORY: {
                ScheduledSyncAction.SYNC.value,
                ScheduledSyncAction.AUDIT.value,
                ScheduledSyncAction.REPAIR.value,
            },
            ScheduledSyncType.FINANCE: {
                ScheduledSyncAction.FULL.value,
                ScheduledSyncAction.QUICK.value,
            },
            ScheduledSyncType.COMPANY: {
                ScheduledSyncAction.FULL.value,
                ScheduledSyncAction.QUICK.value,
            },
        }
        if sync_action not in valid_actions[sync_type_enum]:
            raise SchedulerValidationError("Invalid sync_action for selected sync_type")

        normalized_symbols = [
            str(symbol).strip().upper()[:10]
            for symbol in (payload.get("symbols") or [])
            if str(symbol).strip()
        ]
        normalized_symbols = list(dict.fromkeys(normalized_symbols))

        index_symbol_raw = str(payload.get("index_symbol") or "").strip()
        index_symbol = index_symbol_raw.upper() if index_symbol_raw else None

        starts_at = self._parse_datetime_value(payload.get("starts_at"), "starts_at")
        timezone_name = str(payload.get("timezone") or self.DEFAULT_TIMEZONE).strip() or self.DEFAULT_TIMEZONE
        if timezone_name != self.DEFAULT_TIMEZONE:
            raise SchedulerValidationError("Only Asia/Ho_Chi_Minh timezone is supported")

        interval_value_raw = payload.get("interval_value")
        try:
            interval_value = int(interval_value_raw)
        except (TypeError, ValueError) as exc:
            raise SchedulerValidationError("interval_value must be a positive integer") from exc
        if interval_value <= 0:
            raise SchedulerValidationError("interval_value must be a positive integer")

        interval_unit = str(payload.get("interval_unit") or "").strip().lower()
        try:
            interval_unit_enum = ScheduledSyncIntervalUnit(interval_unit)
        except ValueError as exc:
            raise SchedulerValidationError("Invalid interval_unit") from exc

        max_retries_raw = payload.get("max_retries", 3)
        try:
            max_retries = int(max_retries_raw)
        except (TypeError, ValueError) as exc:
            raise SchedulerValidationError("max_retries must be a non-negative integer") from exc
        if max_retries < 0:
            raise SchedulerValidationError("max_retries must be a non-negative integer")

        partial_success_threshold_raw = payload.get("partial_success_failure_threshold_percent", 10)
        try:
            partial_success_failure_threshold_percent = int(partial_success_threshold_raw)
        except (TypeError, ValueError) as exc:
            raise SchedulerValidationError(
                "partial_success_failure_threshold_percent must be an integer between 0 and 100"
            ) from exc
        if partial_success_failure_threshold_percent < 0 or partial_success_failure_threshold_percent > 100:
            raise SchedulerValidationError(
                "partial_success_failure_threshold_percent must be an integer between 0 and 100"
            )

        date_from = self._parse_date_value(payload.get("date_from")) if payload.get("date_from") else None
        date_to = self._parse_date_value(payload.get("date_to")) if payload.get("date_to") else None
        auto_repair = bool(payload.get("auto_repair", False))

        if sync_type_enum == ScheduledSyncType.HISTORY and sync_action in {
            ScheduledSyncAction.AUDIT.value,
            ScheduledSyncAction.REPAIR.value,
        }:
            if date_from is None or date_to is None:
                raise SchedulerValidationError("date_from and date_to are required for history audit and repair jobs")
            if date_to < date_from:
                raise SchedulerValidationError("date_to must be on or after date_from")
        else:
            if date_from is not None or date_to is not None:
                raise SchedulerValidationError("date_from/date_to are only supported for history audit and repair jobs")
            date_from = None
            date_to = None

        if sync_action != ScheduledSyncAction.AUDIT.value:
            auto_repair = False

        starts_at_utc = to_utc_naive(starts_at)
        next_run_at = self._compute_next_run_at(
            starts_at_utc=starts_at_utc,
            interval_value=interval_value,
            interval_unit=interval_unit_enum.value,
            now_utc=utc_now(),
        )

        return {
            "name": name,
            "enabled": bool(payload.get("enabled", True)),
            "sync_type": sync_type_enum.value,
            "sync_action": sync_action,
            "index_symbol": index_symbol,
            "symbols": normalized_symbols,
            "date_from": date_from,
            "date_to": date_to,
            "auto_repair": auto_repair,
            "starts_at": starts_at_utc,
            "interval_value": interval_value,
            "interval_unit": interval_unit_enum.value,
            "timezone": timezone_name,
            "max_retries": max_retries,
            "partial_success_failure_threshold_percent": partial_success_failure_threshold_percent,
            "next_run_at": next_run_at,
        }

    def _merge_job_payload(self, job: ScheduledSyncJob, payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "name": job.name,
            "enabled": job.enabled,
            "sync_type": job.sync_type,
            "sync_action": job.sync_action,
            "index_symbol": job.index_symbol,
            "symbols": list(job.symbols or []),
            "date_from": job.date_from.isoformat() if job.date_from else None,
            "date_to": job.date_to.isoformat() if job.date_to else None,
            "auto_repair": job.auto_repair,
            "starts_at": utc_naive_to_local_iso(job.starts_at),
            "interval_value": job.interval_value,
            "interval_unit": job.interval_unit,
            "timezone": job.timezone,
            "max_retries": job.max_retries,
            "partial_success_failure_threshold_percent": job.partial_success_failure_threshold_percent,
        }
        merged.update(payload)
        return merged

    def _parse_datetime_value(self, value: Any, field_name: str) -> datetime:
        if isinstance(value, datetime):
            return value
        if not value:
            raise SchedulerValidationError(f"{field_name} is required")
        try:
            return datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise SchedulerValidationError(f"{field_name} must be a valid ISO datetime") from exc

    def _parse_date_value(self, value: Any) -> date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError as exc:
            raise SchedulerValidationError("Dates must use YYYY-MM-DD") from exc

    def _compute_next_run_at(
        self,
        *,
        starts_at_utc: datetime,
        interval_value: int,
        interval_unit: str,
        now_utc: datetime,
    ) -> datetime:
        next_run_at = starts_at_utc
        while next_run_at < now_utc:
            next_run_at = self._advance_utc_occurrence(
                current_utc=next_run_at,
                interval_value=interval_value,
                interval_unit=interval_unit,
            )
        return next_run_at

    def _advance_utc_occurrence(
        self,
        *,
        current_utc: datetime,
        interval_value: int,
        interval_unit: str,
    ) -> datetime:
        if interval_unit == ScheduledSyncIntervalUnit.MINUTES.value:
            delta = timedelta(minutes=interval_value)
        elif interval_unit == ScheduledSyncIntervalUnit.HOURS.value:
            delta = timedelta(hours=interval_value)
        else:
            delta = timedelta(days=interval_value)
        return current_utc + delta

    def _serialize_job(self, job: ScheduledSyncJob) -> dict[str, Any]:
        return {
            "id": job.id,
            "name": job.name,
            "enabled": job.enabled,
            "sync_type": job.sync_type,
            "sync_action": job.sync_action,
            "index_symbol": job.index_symbol,
            "symbols": list(job.symbols or []),
            "date_from": job.date_from.isoformat() if job.date_from else None,
            "date_to": job.date_to.isoformat() if job.date_to else None,
            "auto_repair": job.auto_repair,
            "starts_at": utc_naive_to_local_iso(job.starts_at),
            "interval_value": job.interval_value,
            "interval_unit": job.interval_unit,
            "timezone": job.timezone,
            "max_retries": job.max_retries,
            "partial_success_failure_threshold_percent": job.partial_success_failure_threshold_percent,
            "next_run_at": utc_naive_to_local_iso(job.next_run_at),
            "last_run_at": utc_naive_to_local_iso(job.last_run_at),
            "created_at": utc_naive_to_local_iso(job.created_at),
            "updated_at": utc_naive_to_local_iso(job.updated_at),
        }

    def _serialize_run(self, run: ScheduledSyncJobRun, job: ScheduledSyncJob) -> dict[str, Any]:
        return {
            "id": run.id,
            "job_id": run.job_id,
            "job_name": job.name,
            "sync_type": job.sync_type,
            "sync_action": job.sync_action,
            "attempt_number": run.attempt_number,
            "status": run.status,
            "scheduled_for": utc_naive_to_local_iso(run.scheduled_for),
            "started_at": utc_naive_to_local_iso(run.started_at),
            "finished_at": utc_naive_to_local_iso(run.finished_at),
            "error": run.error,
            "summary": run.summary or {},
            "created_at": utc_naive_to_local_iso(run.created_at),
            "updated_at": utc_naive_to_local_iso(run.updated_at),
        }

    def _classify_completed_run(
        self,
        *,
        job: ScheduledSyncJob,
        summary: Dict[str, Any],
        label: str,
    ) -> SchedulerDispatchResult:
        normalized_summary = dict(summary or {})
        failed_symbols = max(0, int(normalized_summary.get("failed_symbols") or 0))
        total_symbols = self._extract_total_symbols(normalized_summary)
        threshold_percent = int(job.partial_success_failure_threshold_percent or 0)
        failure_ratio = (failed_symbols / total_symbols) if total_symbols > 0 else None

        normalized_summary["total_symbols"] = total_symbols
        normalized_summary["partial_success_failure_threshold_percent"] = threshold_percent
        normalized_summary["failure_ratio"] = failure_ratio

        if failed_symbols == 0:
            return SchedulerDispatchResult(
                status=ScheduledSyncRunStatus.SUCCEEDED,
                summary=normalized_summary,
            )

        if total_symbols <= 0:
            return SchedulerDispatchResult(
                status=ScheduledSyncRunStatus.FAILED,
                summary=normalized_summary,
                error=f"{label} completed with failed symbols but total symbol count was unavailable",
            )

        threshold_ratio = threshold_percent / 100.0
        ratio_percent = failure_ratio * 100.0

        if failure_ratio <= threshold_ratio:
            normalized_summary["message"] = (
                f"{label} partially succeeded: {failed_symbols}/{total_symbols} symbols failed "
                f"({ratio_percent:.2f}% <= {threshold_percent}%)"
            )
            return SchedulerDispatchResult(
                status=ScheduledSyncRunStatus.PARTIAL_SUCCEEDED,
                summary=normalized_summary,
            )

        error = (
            f"{label} failed: {failed_symbols}/{total_symbols} symbols failed "
            f"({ratio_percent:.2f}% > {threshold_percent}%)"
        )
        normalized_summary["message"] = error
        return SchedulerDispatchResult(
            status=ScheduledSyncRunStatus.FAILED,
            summary=normalized_summary,
            error=error,
        )

    def _extract_total_symbols(self, summary: Dict[str, Any]) -> int:
        candidates = (
            summary.get("total_symbols"),
            summary.get("audited_symbols"),
            summary.get("processed_symbols"),
        )
        for candidate in candidates:
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value

        success_symbols = max(0, int(summary.get("success_symbols") or 0))
        failed_symbols = max(0, int(summary.get("failed_symbols") or 0))
        return success_symbols + failed_symbols
