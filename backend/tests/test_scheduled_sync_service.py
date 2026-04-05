from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.db.models import ScheduledSyncJob, ScheduledSyncJobRun
from app.services.sync_status import GlobalSyncStatus
from app.services.vnstock_service.scheduler import (
    ScheduledSyncRunStatus,
    ScheduledSyncService,
    VN_TZ,
    utc_now,
)


def _iso_in_vn(delta: timedelta | None = None) -> str:
    now = datetime.now(VN_TZ)
    if delta is not None:
        now += delta
    return now.isoformat(timespec="seconds")


def _build_service(**overrides) -> ScheduledSyncService:
    async def _default_history_sync(**_kwargs):
        return {"started": True, "message": "started", "state": "running"}

    async def _default_history_audit_sync(**_kwargs):
        return {"started": True, "message": "History audit completed", "failed_symbols": 0}

    async def _default_history_repair_sync(**_kwargs):
        return {"started": True, "message": "Repair sync completed", "failed_symbols": 0}

    async def _default_finance_sync(**_kwargs):
        return {"started": True, "message": "started", "state": "running"}

    async def _default_company_sync(**_kwargs):
        return {"started": True, "message": "started", "state": "running"}

    return ScheduledSyncService(
        run_history_sync=overrides.get("run_history_sync", _default_history_sync),
        run_history_audit_sync=overrides.get("run_history_audit_sync", _default_history_audit_sync),
        run_history_repair_sync=overrides.get("run_history_repair_sync", _default_history_repair_sync),
        run_finance_sync=overrides.get("run_finance_sync", _default_finance_sync),
        run_company_sync=overrides.get("run_company_sync", _default_company_sync),
        poll_interval_seconds=0.05,
        status_poll_seconds=0.01,
        deferred_poll_seconds=0.05,
        retry_base_minutes=15,
    )


@pytest.mark.asyncio
async def test_scheduler_create_job_validates_and_serializes_dates(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "sync_status", GlobalSyncStatus())
    service = _build_service()

    job = await service.create_job(
        {
            "name": "Daily Finance Quick",
            "sync_type": "finance",
            "sync_action": "quick",
            "symbols": ["VCB", "vcb", "ACB"],
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 2,
        }
    )

    assert job["name"] == "Daily Finance Quick"
    assert job["symbols"] == ["VCB", "ACB"]
    assert job["sync_type"] == "finance"
    assert job["sync_action"] == "quick"
    assert job["timezone"] == "Asia/Ho_Chi_Minh"
    assert job["partial_success_failure_threshold_percent"] == 10
    assert job["next_run_at"] is not None


@pytest.mark.asyncio
async def test_scheduler_enqueue_due_runs_advances_next_run_at(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "sync_status", GlobalSyncStatus())
    service = _build_service()
    job_payload = {
        "name": "History Repair",
        "sync_type": "history",
        "sync_action": "repair",
        "symbols": ["AAA"],
        "date_from": "2026-01-01",
        "date_to": "2026-01-05",
        "starts_at": _iso_in_vn(timedelta(hours=1)),
        "interval_value": 1,
        "interval_unit": "hours",
        "timezone": "Asia/Ho_Chi_Minh",
        "max_retries": 1,
    }
    created = await service.create_job(job_payload)

    job = await db_session.get(ScheduledSyncJob, created["id"])
    original_next_run = job.next_run_at
    job.next_run_at = utc_now() - timedelta(minutes=1)
    await db_session.commit()

    await service._enqueue_due_runs()

    refreshed = await db_session.get(ScheduledSyncJob, created["id"])
    await db_session.refresh(refreshed)
    rows = (
        await db_session.execute(
            select(ScheduledSyncJobRun).where(ScheduledSyncJobRun.job_id == created["id"])
        )
    ).scalars().all()

    assert len(rows) == 1
    assert rows[0].status == ScheduledSyncRunStatus.QUEUED.value
    assert refreshed.next_run_at > utc_now()
    assert refreshed.next_run_at != original_next_run


@pytest.mark.asyncio
async def test_scheduler_minute_interval_advances_by_minutes(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "sync_status", GlobalSyncStatus())
    service = _build_service()
    created = await service.create_job(
        {
            "name": "Finance Every 5 Minutes",
            "sync_type": "finance",
            "sync_action": "quick",
            "starts_at": _iso_in_vn(timedelta(minutes=10)),
            "interval_value": 5,
            "interval_unit": "minutes",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
        }
    )

    job = await db_session.get(ScheduledSyncJob, created["id"])
    original_next_run = job.next_run_at
    advanced = service._advance_utc_occurrence(
        current_utc=job.next_run_at,
        interval_value=job.interval_value,
        interval_unit=job.interval_unit,
    )

    assert (advanced - original_next_run) == timedelta(minutes=5)


@pytest.mark.asyncio
async def test_scheduler_retry_creates_followup_attempt_until_max_retries(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    fresh_status = GlobalSyncStatus()
    monkeypatch.setattr(scheduler_module, "sync_status", fresh_status)

    async def _failing_history_audit(**_kwargs):
        return {
            "started": True,
            "message": "History audit completed with failures",
            "failed_symbols": 1,
        }

    service = _build_service(run_history_audit_sync=_failing_history_audit)
    created = await service.create_job(
        {
            "name": "Audit Job",
            "sync_type": "history",
            "sync_action": "audit",
            "symbols": ["AAA"],
            "date_from": "2026-01-01",
            "date_to": "2026-01-02",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 1,
        }
    )

    first_run = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.QUEUED.value,
        scheduled_for=utc_now(),
        summary={},
    )
    db_session.add(first_run)
    await db_session.commit()

    await service._run_queued_job(first_run.id)

    runs = (
        await db_session.execute(
            select(ScheduledSyncJobRun)
            .where(ScheduledSyncJobRun.job_id == created["id"])
            .order_by(ScheduledSyncJobRun.attempt_number.asc())
        )
    ).scalars().all()
    for row in runs:
        await db_session.refresh(row)

    assert len(runs) == 2
    assert runs[0].status == ScheduledSyncRunStatus.FAILED.value
    assert runs[1].status == ScheduledSyncRunStatus.QUEUED.value
    assert runs[1].attempt_number == 2

    await service._run_queued_job(runs[1].id)

    final_runs = (
        await db_session.execute(
            select(ScheduledSyncJobRun)
            .where(ScheduledSyncJobRun.job_id == created["id"])
            .order_by(ScheduledSyncJobRun.attempt_number.asc())
        )
    ).scalars().all()
    for row in final_runs:
        await db_session.refresh(row)
    assert len(final_runs) == 2
    assert final_runs[-1].status == ScheduledSyncRunStatus.FAILED.value


@pytest.mark.asyncio
async def test_scheduler_waits_for_existing_runtime_before_starting_next_job(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    fresh_status = GlobalSyncStatus()
    fresh_status.start_finance_sync(total_symbols=1)
    monkeypatch.setattr(scheduler_module, "sync_status", fresh_status)

    called = {"count": 0}

    async def _history_repair(**_kwargs):
        called["count"] += 1
        return {"started": True, "message": "Repair sync completed", "failed_symbols": 0}

    service = _build_service(run_history_repair_sync=_history_repair)
    created = await service.create_job(
        {
            "name": "Repair Job",
            "sync_type": "history",
            "sync_action": "repair",
            "symbols": ["AAA"],
            "date_from": "2026-01-01",
            "date_to": "2026-01-03",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
        }
    )

    queued = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.QUEUED.value,
        scheduled_for=utc_now(),
        summary={},
    )
    db_session.add(queued)
    await db_session.commit()

    await service._maybe_start_next_run()
    assert service._current_run_task is None
    assert called["count"] == 0

    fresh_status.complete_finance_sync(success=True)
    await service._maybe_start_next_run()
    assert service._current_run_task is not None
    await asyncio.gather(service._current_run_task)
    assert called["count"] == 1


@pytest.mark.asyncio
async def test_scheduler_dispatches_background_finance_and_records_success(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    fresh_status = GlobalSyncStatus()
    monkeypatch.setattr(scheduler_module, "sync_status", fresh_status)

    captured = {}

    async def _finance_sync(**kwargs):
        captured.update(kwargs)
        fresh_status.start_finance_sync(total_symbols=2)

        async def _complete() -> None:
            await asyncio.sleep(0.02)
            fresh_status.update_finance_sync_progress(
                processed_symbols=2,
                success_symbols=2,
                failed_symbols=0,
                current_symbol=None,
            )
            fresh_status.complete_finance_sync(success=True)

        asyncio.create_task(_complete())
        return {"started": True, "message": "Finance sync started", "state": "running"}

    service = _build_service(run_finance_sync=_finance_sync)
    created = await service.create_job(
        {
            "name": "Finance Quick",
            "sync_type": "finance",
            "sync_action": "quick",
            "symbols": ["VCB"],
            "index_symbol": "VN30",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
        }
    )

    queued = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.QUEUED.value,
        scheduled_for=utc_now(),
        summary={},
    )
    db_session.add(queued)
    await db_session.commit()

    await service._run_queued_job(queued.id)

    refreshed = await db_session.get(ScheduledSyncJobRun, queued.id)
    await db_session.refresh(refreshed)
    assert refreshed.status == ScheduledSyncRunStatus.SUCCEEDED.value
    assert refreshed.summary["processed_symbols"] == 2
    assert captured["quick_sync"] is True
    assert captured["index_symbol"] == "VN30"
    assert captured["symbols"] == ["VCB"]


@pytest.mark.asyncio
async def test_scheduler_marks_background_finance_partial_success_within_threshold(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    fresh_status = GlobalSyncStatus()
    monkeypatch.setattr(scheduler_module, "sync_status", fresh_status)

    async def _finance_sync(**_kwargs):
        fresh_status.start_finance_sync(total_symbols=10)

        async def _complete() -> None:
            await asyncio.sleep(0.02)
            fresh_status.update_finance_sync_progress(
                processed_symbols=10,
                success_symbols=9,
                failed_symbols=1,
                current_symbol=None,
                failed_tickers=["BBB"],
            )
            fresh_status.complete_finance_sync(
                success=True,
                error="Finance sync completed with 1 failed symbols",
            )

        asyncio.create_task(_complete())
        return {"started": True, "message": "Finance sync started", "state": "running"}

    service = _build_service(run_finance_sync=_finance_sync)
    created = await service.create_job(
        {
            "name": "Finance Partial",
            "sync_type": "finance",
            "sync_action": "quick",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 2,
            "partial_success_failure_threshold_percent": 10,
        }
    )

    queued = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.QUEUED.value,
        scheduled_for=utc_now(),
        summary={},
    )
    db_session.add(queued)
    await db_session.commit()

    await service._run_queued_job(queued.id)

    runs = (
        await db_session.execute(
            select(ScheduledSyncJobRun)
            .where(ScheduledSyncJobRun.job_id == created["id"])
            .order_by(ScheduledSyncJobRun.attempt_number.asc())
        )
    ).scalars().all()
    for row in runs:
        await db_session.refresh(row)

    assert len(runs) == 1
    assert runs[0].status == ScheduledSyncRunStatus.PARTIAL_SUCCEEDED.value
    assert runs[0].error is None
    assert runs[0].summary["partial_success_failure_threshold_percent"] == 10
    assert runs[0].summary["failure_ratio"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_scheduler_marks_audit_run_partial_success_within_threshold(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "sync_status", GlobalSyncStatus())

    async def _history_audit(**_kwargs):
        return {
            "started": True,
            "message": "History audit completed",
            "processed_symbols": 10,
            "success_symbols": 9,
            "failed_symbols": 1,
            "audited_symbols": 10,
            "results": [],
        }

    service = _build_service(run_history_audit_sync=_history_audit)
    created = await service.create_job(
        {
            "name": "Audit Partial",
            "sync_type": "history",
            "sync_action": "audit",
            "symbols": ["AAA"],
            "date_from": "2026-01-01",
            "date_to": "2026-01-02",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 1,
            "partial_success_failure_threshold_percent": 10,
        }
    )

    queued = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.QUEUED.value,
        scheduled_for=utc_now(),
        summary={},
    )
    db_session.add(queued)
    await db_session.commit()

    await service._run_queued_job(queued.id)

    runs = (
        await db_session.execute(
            select(ScheduledSyncJobRun)
            .where(ScheduledSyncJobRun.job_id == created["id"])
            .order_by(ScheduledSyncJobRun.attempt_number.asc())
        )
    ).scalars().all()
    for row in runs:
        await db_session.refresh(row)

    assert len(runs) == 1
    assert runs[0].status == ScheduledSyncRunStatus.PARTIAL_SUCCEEDED.value
    assert runs[0].summary["failure_ratio"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_scheduler_above_threshold_failures_still_retry(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "sync_status", GlobalSyncStatus())

    async def _history_repair(**_kwargs):
        return {
            "started": True,
            "message": "Repair sync completed",
            "processed_symbols": 10,
            "success_symbols": 8,
            "failed_symbols": 2,
            "results": [],
        }

    service = _build_service(run_history_repair_sync=_history_repair)
    created = await service.create_job(
        {
            "name": "Repair Fails",
            "sync_type": "history",
            "sync_action": "repair",
            "symbols": ["AAA"],
            "date_from": "2026-01-01",
            "date_to": "2026-01-02",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 1,
            "partial_success_failure_threshold_percent": 10,
        }
    )

    queued = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.QUEUED.value,
        scheduled_for=utc_now(),
        summary={},
    )
    db_session.add(queued)
    await db_session.commit()

    await service._run_queued_job(queued.id)

    runs = (
        await db_session.execute(
            select(ScheduledSyncJobRun)
            .where(ScheduledSyncJobRun.job_id == created["id"])
            .order_by(ScheduledSyncJobRun.attempt_number.asc())
        )
    ).scalars().all()
    for row in runs:
        await db_session.refresh(row)

    assert len(runs) == 2
    assert runs[0].status == ScheduledSyncRunStatus.FAILED.value
    assert runs[0].error is not None
    assert "20.00% > 10%" in runs[0].error
    assert runs[1].status == ScheduledSyncRunStatus.QUEUED.value


@pytest.mark.asyncio
async def test_scheduler_runtime_exception_without_summary_marks_failed(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "sync_status", GlobalSyncStatus())

    async def _history_audit(**_kwargs):
        raise RuntimeError("upstream exploded")

    service = _build_service(run_history_audit_sync=_history_audit)
    created = await service.create_job(
        {
            "name": "Audit Crash",
            "sync_type": "history",
            "sync_action": "audit",
            "symbols": ["AAA"],
            "date_from": "2026-01-01",
            "date_to": "2026-01-02",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
        }
    )

    queued = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.QUEUED.value,
        scheduled_for=utc_now(),
        summary={},
    )
    db_session.add(queued)
    await db_session.commit()

    await service._run_queued_job(queued.id)

    refreshed = await db_session.get(ScheduledSyncJobRun, queued.id)
    await db_session.refresh(refreshed)
    assert refreshed.status == ScheduledSyncRunStatus.FAILED.value
    assert refreshed.error == "upstream exploded"


@pytest.mark.asyncio
async def test_scheduler_recover_incomplete_runs_marks_running_rows_failed(db_session, monkeypatch):
    from app.services.vnstock_service import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "sync_status", GlobalSyncStatus())
    service = _build_service()
    created = await service.create_job(
        {
            "name": "Company Full",
            "sync_type": "company",
            "sync_action": "full",
            "starts_at": _iso_in_vn(timedelta(hours=1)),
            "interval_value": 1,
            "interval_unit": "days",
            "timezone": "Asia/Ho_Chi_Minh",
            "max_retries": 0,
        }
    )

    running = ScheduledSyncJobRun(
        job_id=created["id"],
        attempt_number=1,
        status=ScheduledSyncRunStatus.RUNNING.value,
        scheduled_for=utc_now(),
        started_at=utc_now(),
        summary={},
    )
    db_session.add(running)
    await db_session.commit()

    await service.recover_incomplete_runs()

    refreshed = await db_session.get(ScheduledSyncJobRun, running.id)
    await db_session.refresh(refreshed)
    assert refreshed.status == ScheduledSyncRunStatus.FAILED.value
    assert "restarted" in refreshed.error.lower()
