from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from tools.cafef_scraper.catalog import Catalog
from tools.cafef_scraper.classifier import (
    classify_doc_type,
    extract_event_date,
    likely_target_article_url,
    normalize_text,
    slugify,
)
from tools.cafef_scraper.discovery import (
    CafefDiscovery,
    build_symbol_probe_url,
)
from tools.cafef_scraper.downloader import HttpDownloader, RateLimiter
from tools.cafef_scraper.extractor import parse_article_detail
from tools.cafef_scraper.types import CoverageStatus, DocStatus, DocType, DocumentCandidate, SourceType

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


@dataclass(slots=True)
class RunnerOptions:
    output_dir: Path
    max_concurrency: int = 4
    rate_limit_rps: float = 1.5
    adaptive_rate_limit: bool = False
    adaptive_min_rps: float = 0.5
    adaptive_recovery_multiplier: float = 1.05
    adaptive_cooldown_seconds: float = 20.0
    adaptive_cooldown_streak: int = 3
    dry_run: bool = False
    max_retries: int = 5
    timeout_seconds: float = 30.0
    id_scan_probe_max_retries: int | None = None
    id_scan_probe_timeout_seconds: float | None = None
    symbol: str = "E1VFVN30"


class CafefScraperRunner:
    def __init__(self, options: RunnerOptions, logger: logging.Logger):
        if options.id_scan_probe_max_retries is not None and options.id_scan_probe_max_retries < 1:
            raise ValueError("--id-scan-probe-max-retries must be >= 1.")
        if options.id_scan_probe_timeout_seconds is not None and options.id_scan_probe_timeout_seconds < 1.0:
            raise ValueError("--id-scan-probe-timeout-seconds must be >= 1.0.")
        self.options = options
        self.logger = logger
        self.output_dir = options.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = Catalog(self.output_dir / "catalog.sqlite")
        self._rate_limiter = RateLimiter(
            rate_limit_rps=options.rate_limit_rps,
            adaptive=options.adaptive_rate_limit,
            min_rps=options.adaptive_min_rps,
            recovery_multiplier=options.adaptive_recovery_multiplier,
            cooldown_seconds=options.adaptive_cooldown_seconds,
            cooldown_trigger_streak=options.adaptive_cooldown_streak,
            logger=self.logger,
        )
        self._client = httpx.AsyncClient()
        self._downloader = HttpDownloader(
            client=self._client,
            limiter=self._rate_limiter,
            max_retries=options.max_retries,
            timeout_seconds=options.timeout_seconds,
        )
        self._discovery = CafefDiscovery(
            fetch_text=self._fetch_text,
            symbol=options.symbol,
            logger=self.logger,
        )
        self._current_run_id: int | None = None

    async def close(self) -> None:
        await self._client.aclose()
        self.catalog.close()

    async def run_backfill(
        self,
        start_date: date,
        end_date: date,
        doc_types: set[DocType],
        resume: bool = True,
        reset_discovery_in_resume: bool = False,
        mode: str = "full",
        discovery_strategy: str = "balanced",
        max_event_pages: int = 350,
        max_sitemaps: int | None = None,
        event_feed_enabled: bool = True,
        sitemap_enabled: bool = True,
        id_range: tuple[int, int] | None = None,
        id_scan_coarse_step: int = 250,
        id_scan_coarse_offsets: int = 1,
        id_scan_window: int = 800,
        id_scan_coarse_only: bool = False,
        retry_failed_enabled: bool = True,
    ) -> dict[str, Any]:
        if mode not in {"full", "discover-sources", "discover-idscan", "consume-only"}:
            raise ValueError(f"Unsupported mode '{mode}'")
        if mode == "discover-idscan" and id_range is None:
            raise ValueError("--mode discover-idscan requires --id-range START:END")
        if id_scan_coarse_only and id_range is None:
            raise ValueError("--id-scan-coarse-only requires --id-range START:END")
        if mode == "consume-only" and (not resume or reset_discovery_in_resume):
            raise ValueError("--mode consume-only requires --resume and no discovery reset")
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "doc_types": sorted([d.value for d in doc_types]),
            "resume": resume,
            "reset_discovery_in_resume": reset_discovery_in_resume or (not resume),
            "mode": mode,
            "discovery_strategy": discovery_strategy,
            "max_event_pages": max_event_pages,
            "max_sitemaps": max_sitemaps,
            "event_feed_enabled": event_feed_enabled,
            "sitemap_enabled": sitemap_enabled,
            "id_range": list(id_range) if id_range else None,
            "id_scan_coarse_step": id_scan_coarse_step,
            "id_scan_coarse_offsets": id_scan_coarse_offsets,
            "id_scan_window": id_scan_window,
            "id_scan_coarse_only": id_scan_coarse_only,
            "retry_failed_enabled": retry_failed_enabled,
            "dry_run": self.options.dry_run,
            "id_scan_probe_max_retries": self.options.id_scan_probe_max_retries,
            "id_scan_probe_timeout_seconds": self.options.id_scan_probe_timeout_seconds,
        }
        run_id = self.catalog.start_run("backfill", params)
        self._current_run_id = run_id
        try:
            effective_reset_discovery = (reset_discovery_in_resume or (not resume)) and mode != "consume-only"
            if not resume and mode != "consume-only":
                self.catalog.reset_range(start_date=start_date, end_date=end_date, doc_types=doc_types)
            stats = await self._run_range(
                mode="backfill",
                start_date=start_date,
                end_date=end_date,
                doc_types=doc_types,
                pipeline_mode=mode,
                discovery_strategy=discovery_strategy,
                reset_discovery=effective_reset_discovery,
                max_event_pages=max_event_pages,
                max_sitemaps=max_sitemaps,
                event_feed_enabled=event_feed_enabled,
                sitemap_enabled=sitemap_enabled,
                id_range=id_range,
                id_scan_coarse_step=id_scan_coarse_step,
                id_scan_coarse_offsets=id_scan_coarse_offsets,
                id_scan_window=id_scan_window,
                id_scan_coarse_only=id_scan_coarse_only,
                retry_failed=retry_failed_enabled,
            )
            self.catalog.set_state("last_backfill_to_date", end_date.isoformat())
            self.catalog.finish_run(run_id, "SUCCESS", stats=stats)
            return stats
        except Exception as exc:
            self.catalog.finish_run(run_id, "FAILED", error=str(exc))
            raise
        finally:
            self._current_run_id = None

    async def run_incremental(
        self,
        lookback_days: int,
        doc_types: set[DocType],
        reset_discovery_in_resume: bool = False,
        mode: str = "full",
        discovery_strategy: str = "balanced",
        max_event_pages: int = 120,
        max_sitemaps: int = 350,
        event_feed_enabled: bool = True,
        sitemap_enabled: bool = True,
        id_range: tuple[int, int] | None = None,
        id_scan_coarse_step: int = 250,
        id_scan_coarse_offsets: int = 1,
        id_scan_window: int = 800,
        id_scan_coarse_only: bool = False,
        retry_failed_enabled: bool = True,
    ) -> dict[str, Any]:
        if mode not in {"full", "discover-sources", "discover-idscan", "consume-only"}:
            raise ValueError(f"Unsupported mode '{mode}'")
        if mode == "discover-idscan" and id_range is None:
            raise ValueError("--mode discover-idscan requires --id-range START:END")
        if id_scan_coarse_only and id_range is None:
            raise ValueError("--id-scan-coarse-only requires --id-range START:END")
        end_date = datetime.now(tz=VN_TZ).date()
        start_date = end_date - timedelta(days=max(1, lookback_days))
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lookback_days": lookback_days,
            "doc_types": sorted([d.value for d in doc_types]),
            "reset_discovery_in_resume": reset_discovery_in_resume,
            "mode": mode,
            "discovery_strategy": discovery_strategy,
            "max_event_pages": max_event_pages,
            "max_sitemaps": max_sitemaps,
            "event_feed_enabled": event_feed_enabled,
            "sitemap_enabled": sitemap_enabled,
            "id_range": list(id_range) if id_range else None,
            "id_scan_coarse_step": id_scan_coarse_step,
            "id_scan_coarse_offsets": id_scan_coarse_offsets,
            "id_scan_window": id_scan_window,
            "id_scan_coarse_only": id_scan_coarse_only,
            "retry_failed_enabled": retry_failed_enabled,
            "dry_run": self.options.dry_run,
            "id_scan_probe_max_retries": self.options.id_scan_probe_max_retries,
            "id_scan_probe_timeout_seconds": self.options.id_scan_probe_timeout_seconds,
        }
        run_id = self.catalog.start_run("incremental", params)
        self._current_run_id = run_id
        try:
            stats = await self._run_range(
                mode="incremental",
                start_date=start_date,
                end_date=end_date,
                doc_types=doc_types,
                pipeline_mode=mode,
                discovery_strategy=discovery_strategy,
                reset_discovery=reset_discovery_in_resume,
                max_event_pages=max_event_pages,
                max_sitemaps=max_sitemaps,
                event_feed_enabled=event_feed_enabled,
                sitemap_enabled=sitemap_enabled,
                id_range=id_range,
                id_scan_coarse_step=id_scan_coarse_step,
                id_scan_coarse_offsets=id_scan_coarse_offsets,
                id_scan_window=id_scan_window,
                id_scan_coarse_only=id_scan_coarse_only,
                retry_failed=retry_failed_enabled,
            )
            self.catalog.set_state("last_incremental_sync_date", end_date.isoformat())
            self.catalog.finish_run(run_id, "SUCCESS", stats=stats)
            return stats
        except Exception as exc:
            self.catalog.finish_run(run_id, "FAILED", error=str(exc))
            raise
        finally:
            self._current_run_id = None

    async def run_retry_failed(
        self,
        start_date: date,
        end_date: date,
        doc_types: set[DocType],
        limit: int = 300,
    ) -> dict[str, Any]:
        run_id = self.catalog.start_run(
            "retry-failed",
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "doc_types": sorted([d.value for d in doc_types]),
                "limit": limit,
                "dry_run": self.options.dry_run,
            },
        )
        self._current_run_id = run_id
        try:
            stats = {
                "retried_failed_documents": 0,
                "retried_failed_success": 0,
                "retry_failed_skipped_url_guess": 0,
            }
            failed_rows = self.catalog.list_failed_documents(start_date=start_date, end_date=end_date, limit=limit)
            for row in failed_rows:
                doc_type = DocType(str(row["doc_type"]))
                if doc_type not in doc_types:
                    continue
                source_type = str(row["source_type"])
                if source_type == SourceType.URL_GUESS.value:
                    stats["retry_failed_skipped_url_guess"] += 1
                    continue
                stats["retried_failed_documents"] += 1
                if await self._retry_document_row(row):
                    stats["retried_failed_success"] += 1

            self.catalog.finish_run(run_id, "SUCCESS", stats=stats)
            return stats
        except Exception as exc:
            self.catalog.finish_run(run_id, "FAILED", error=str(exc))
            raise
        finally:
            self._current_run_id = None

    def run_audit(
        self,
        start_date: date,
        end_date: date,
        export_report: Path | None = None,
    ) -> dict[str, Any]:
        rows = self.catalog.list_coverage_rows(start_date, end_date)
        summary = {
            "from_date": start_date.isoformat(),
            "to_date": end_date.isoformat(),
            "total_days": len(rows),
            "both_found": 0,
            "only_basket_found": 0,
            "only_swap_end_found": 0,
            "none_found": 0,
            "has_failed": 0,
        }

        for row in rows:
            basket = str(row["basket_status"])
            swap_end = str(row["swap_end_status"])
            basket_found = basket == CoverageStatus.FOUND.value
            swap_found = swap_end == CoverageStatus.FOUND.value
            has_failed = basket == CoverageStatus.FAILED.value or swap_end == CoverageStatus.FAILED.value
            if has_failed:
                summary["has_failed"] += 1
            if basket_found and swap_found:
                summary["both_found"] += 1
            elif basket_found:
                summary["only_basket_found"] += 1
            elif swap_found:
                summary["only_swap_end_found"] += 1
            else:
                summary["none_found"] += 1

        if export_report:
            export_report.parent.mkdir(parents=True, exist_ok=True)
            with export_report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "basket_status", "swap_end_status", "note"])
                for row in rows:
                    writer.writerow(
                        [
                            row["date"],
                            row["basket_status"],
                            row["swap_end_status"],
                            row["note"] or "",
                        ]
                    )
        return summary

    async def _run_range(
        self,
        mode: str,
        start_date: date,
        end_date: date,
        doc_types: set[DocType],
        pipeline_mode: str,
        discovery_strategy: str,
        reset_discovery: bool,
        max_event_pages: int,
        max_sitemaps: int | None,
        event_feed_enabled: bool,
        sitemap_enabled: bool,
        id_range: tuple[int, int] | None,
        id_scan_coarse_step: int,
        id_scan_coarse_offsets: int,
        id_scan_window: int,
        id_scan_coarse_only: bool,
        retry_failed: bool,
    ) -> dict[str, Any]:
        if id_scan_coarse_step < 1:
            raise ValueError("--id-scan-coarse-step must be >= 1.")
        if id_scan_coarse_offsets < 1:
            raise ValueError("--id-scan-coarse-offsets must be >= 1.")
        if id_scan_window < 0:
            raise ValueError("--id-scan-window must be >= 0.")

        run_source_discovery = pipeline_mode in {"full", "discover-sources"}
        run_id_scan_discovery = pipeline_mode in {"full", "discover-idscan"}
        run_consume = pipeline_mode in {"full", "consume-only"}

        self.catalog.ensure_coverage_range(start_date, end_date)
        stats: dict[str, Any] = {
            "pipeline_mode": pipeline_mode,
            "discovery_scope": "",
            "discovery_strategy": discovery_strategy,
            "event_feed_enabled": event_feed_enabled,
            "sitemap_enabled": sitemap_enabled,
            "discovery_resumed_from_page": 1,
            "discovery_resumed_from_sitemap_index": 0,
            "discovery_resumed_from_id_scan_phase": "",
            "articles_discovered": 0,
            "articles_discovered_event": 0,
            "articles_discovered_sitemap": 0,
            "articles_discovered_id_scan": 0,
            "id_scan_probed": 0,
            "id_scan_coarse_offsets": id_scan_coarse_offsets,
            "id_scan_coarse_only": id_scan_coarse_only,
            "id_scan_suggested_windows": [],
            "articles_pending_queue": 0,
            "articles_fetched": 0,
            "article_fetch_failed": 0,
            "article_filtered_out": 0,
            "documents_discovered": 0,
            "documents_downloaded": 0,
            "documents_failed": 0,
            "documents_skipped_existing": 0,
            "documents_skipped_duplicate": 0,
            "retry_failed_attempted": 0,
            "retry_failed_recovered": 0,
            "retry_failed_skipped_url_guess": 0,
        }
        scope_key = self._build_scope_key(mode=mode, start_date=start_date, end_date=end_date)
        stats["discovery_scope"] = scope_key

        event_done_key = f"discovery:{scope_key}:event_done"
        event_next_page_key = f"discovery:{scope_key}:event_next_page"
        sitemap_done_key = f"discovery:{scope_key}:sitemap_done"
        sitemap_next_index_key = f"discovery:{scope_key}:sitemap_next_index"
        id_scan_phase_key: str | None = None
        id_scan_done_key: str | None = None
        id_scan_coarse_next_key: str | None = None
        id_scan_coarse_offsets_state_key: str | None = None
        id_scan_coarse_hits_key: str | None = None
        id_scan_fine_windows_key: str | None = None
        id_scan_fine_window_index_key: str | None = None
        id_scan_fine_next_id_key: str | None = None
        id_scan_config_key: str | None = None
        if id_range is not None:
            id_scan_scope_suffix = f"{id_range[0]}-{id_range[1]}"
            id_scan_phase_key = f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:phase"
            id_scan_done_key = f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:done"
            id_scan_coarse_next_key = (
                f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:coarse_next_id"
            )
            id_scan_coarse_offsets_state_key = (
                f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:coarse_offsets_state"
            )
            id_scan_coarse_hits_key = f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:coarse_hits"
            id_scan_fine_windows_key = (
                f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:fine_windows"
            )
            id_scan_fine_window_index_key = (
                f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:fine_window_index"
            )
            id_scan_fine_next_id_key = f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:fine_next_id"
            id_scan_config_key = f"discovery:{scope_key}:id_scan:{id_scan_scope_suffix}:config"

        if reset_discovery:
            self.logger.info("Reset discovery checkpoint enabled: resetting scope %s", scope_key)
            self.catalog.reset_discovery_scope(scope_key)

        event_done = self.catalog.get_state(event_done_key) == "1"
        sitemap_done = self.catalog.get_state(sitemap_done_key) == "1"
        event_start_page = int(self.catalog.get_state(event_next_page_key) or "1")
        sitemap_start_index = int(self.catalog.get_state(sitemap_next_index_key) or "0")
        id_scan_phase = self.catalog.get_state(id_scan_phase_key) if id_scan_phase_key else ""
        if id_range is not None and not id_scan_phase:
            id_scan_phase = "coarse"
        stats["discovery_resumed_from_page"] = event_start_page
        stats["discovery_resumed_from_sitemap_index"] = sitemap_start_index
        stats["discovery_resumed_from_id_scan_phase"] = id_scan_phase

        discovered_event_count = 0
        discovered_sitemap_count = 0
        discovered_id_scan_count = 0
        id_scan_probed_count = 0

        self.logger.info(
            "Running range %s to %s (scope=%s mode=%s strategy=%s)",
            start_date.isoformat(),
            end_date.isoformat(),
            scope_key,
            pipeline_mode,
            discovery_strategy,
        )

        if not run_source_discovery:
            self.logger.info("Discovery[event_feed] skipped by mode=%s", pipeline_mode)
            self.logger.info("Discovery[sitemap] skipped by mode=%s", pipeline_mode)
        elif id_scan_coarse_only and id_range is not None:
            self.logger.info("Discovery[event_feed] skipped in id-scan coarse-only mode")
            self.logger.info("Discovery[sitemap] skipped in id-scan coarse-only mode")
        else:
            if not event_feed_enabled:
                self.logger.info("Discovery[event_feed] skipped by configuration")
            elif not event_done:
                self.logger.info("Discovery[event_feed] starting from page=%s", event_start_page)

                def _on_event_page(page_index: int, urls: list[str]) -> None:
                    nonlocal discovered_event_count
                    inserted = self.catalog.enqueue_article_urls(
                        scope_key=scope_key,
                        source_type=SourceType.EVENT_FEED,
                        urls=urls,
                    )
                    discovered_event_count += inserted
                    self.catalog.set_state(event_next_page_key, str(page_index + 1))

                await self._discovery.fetch_event_feed_urls(
                    max_pages=max_event_pages,
                    start_page=event_start_page,
                    on_page=_on_event_page,
                )
                self.catalog.set_state(event_done_key, "1")
            else:
                self.logger.info("Discovery[event_feed] already completed for scope=%s; skipping", scope_key)

            if not sitemap_enabled:
                self.logger.info("Discovery[sitemap] skipped by configuration")
            elif not sitemap_done:
                self.logger.info("Discovery[sitemap] starting from index=%s", sitemap_start_index)

                def _on_sitemap(index: int, _total: int, urls: list[str]) -> None:
                    nonlocal discovered_sitemap_count
                    inserted = self.catalog.enqueue_article_urls(
                        scope_key=scope_key,
                        source_type=SourceType.SITEMAP,
                        urls=urls,
                    )
                    discovered_sitemap_count += inserted
                    self.catalog.set_state(sitemap_next_index_key, str(index))

                await self._discovery.fetch_sitemap_article_urls(
                    start_date=start_date,
                    end_date=end_date,
                    max_sitemaps=max_sitemaps,
                    start_index=sitemap_start_index,
                    on_sitemap=_on_sitemap,
                )
                self.catalog.set_state(sitemap_done_key, "1")
            else:
                self.logger.info("Discovery[sitemap] already completed for scope=%s; skipping", scope_key)

        if not run_id_scan_discovery:
            self.logger.info("Discovery[id_scan] skipped by mode=%s", pipeline_mode)
        elif id_range is not None:
            assert (
                id_scan_done_key
                and id_scan_phase_key
                and id_scan_coarse_next_key
                and id_scan_coarse_offsets_state_key
                and id_scan_coarse_hits_key
                and id_scan_fine_windows_key
                and id_scan_fine_window_index_key
                and id_scan_fine_next_id_key
                and id_scan_config_key
            )
            self._ensure_id_scan_resume_config(
                key=id_scan_config_key,
                coarse_step=id_scan_coarse_step,
                coarse_offsets=id_scan_coarse_offsets,
                window=id_scan_window,
            )
            if self.catalog.get_state(id_scan_done_key) == "1":
                self.logger.info(
                    "Discovery[id_scan] already completed for scope=%s id_range=%s:%s; skipping",
                    scope_key,
                    id_range[0],
                    id_range[1],
                )
            else:
                self.logger.info(
                    "Discovery[id_scan] starting scope=%s id_range=%s:%s coarse_step=%s coarse_offsets=%s window=%s phase=%s",
                    scope_key,
                    id_range[0],
                    id_range[1],
                    id_scan_coarse_step,
                    id_scan_coarse_offsets,
                    id_scan_window,
                    id_scan_phase,
                )
                id_scan_stats = await self._discover_with_id_scan(
                    scope_key=scope_key,
                    start_id=id_range[0],
                    end_id=id_range[1],
                    coarse_step=id_scan_coarse_step,
                    coarse_offsets=id_scan_coarse_offsets,
                    window=id_scan_window,
                    coarse_only=id_scan_coarse_only,
                    phase_key=id_scan_phase_key,
                    done_key=id_scan_done_key,
                    coarse_next_key=id_scan_coarse_next_key,
                    coarse_offsets_state_key=id_scan_coarse_offsets_state_key,
                    coarse_hits_key=id_scan_coarse_hits_key,
                    fine_windows_key=id_scan_fine_windows_key,
                    fine_window_index_key=id_scan_fine_window_index_key,
                    fine_next_id_key=id_scan_fine_next_id_key,
                )
                discovered_id_scan_count = id_scan_stats["new_urls"]
                id_scan_probed_count = id_scan_stats["probed"]
                stats["id_scan_suggested_windows"] = id_scan_stats.get("suggested_windows", [])
        elif run_id_scan_discovery:
            self.logger.info("Discovery[id_scan] skipped because no --id-range was provided")

        stats["articles_discovered_event"] = discovered_event_count
        stats["articles_discovered_sitemap"] = discovered_sitemap_count
        stats["articles_discovered_id_scan"] = discovered_id_scan_count
        stats["id_scan_probed"] = id_scan_probed_count
        stats["articles_discovered"] = (
            discovered_event_count + discovered_sitemap_count + discovered_id_scan_count
        )

        if id_scan_coarse_only and id_range is not None:
            pending_rows = self.catalog.list_pending_article_queue(
                scope_key=scope_key,
                include_filtered=(discovery_strategy == "exhaustive"),
            )
            stats["articles_pending_queue"] = len(pending_rows)
            self.logger.info(
                "ID-scan coarse-only mode complete for scope=%s pending_article_fetch=%s suggested_windows=%s",
                scope_key,
                stats["articles_pending_queue"],
                len(stats["id_scan_suggested_windows"]),
            )
            self.logger.debug("Retry-failed phase skipped by configuration (coarse-only mode)")
            return stats

        if not run_consume:
            pending_rows = self.catalog.list_pending_article_queue(
                scope_key=scope_key,
                include_filtered=(discovery_strategy == "exhaustive"),
            )
            stats["articles_pending_queue"] = len(pending_rows)
            self.logger.info(
                "Consume phase skipped by mode=%s scope=%s pending_article_fetch=%s",
                pipeline_mode,
                scope_key,
                stats["articles_pending_queue"],
            )
            return stats

        pending_rows = self.catalog.list_pending_article_queue(
            scope_key=scope_key,
            include_filtered=(discovery_strategy == "exhaustive"),
        )
        stats["articles_pending_queue"] = len(pending_rows)
        self.logger.info(
            "Discovery complete for scope=%s new_urls=%s pending_article_fetch=%s",
            scope_key,
            stats["articles_discovered"],
            stats["articles_pending_queue"],
        )

        article_docs = await self._process_article_queue(
            queue_rows=pending_rows,
            start_date=start_date,
            end_date=end_date,
            doc_types=doc_types,
            discovery_strategy=discovery_strategy,
        )
        stats["articles_fetched"] = article_docs["articles_fetched"]
        stats["article_fetch_failed"] = article_docs["article_fetch_failed"]
        stats["article_filtered_out"] = article_docs["article_filtered_out"]

        unique_candidates: dict[tuple[str, str, str], tuple[int, DocumentCandidate]] = {}
        for doc_id, candidate in article_docs["candidates"]:
            key = (
                candidate.doc_type.value,
                candidate.event_date.isoformat(),
                candidate.pdf_url,
            )
            if key not in unique_candidates:
                unique_candidates[key] = (doc_id, candidate)
        stats["documents_discovered"] = len(unique_candidates)

        download_stats = await self._process_document_downloads(
            list(unique_candidates.values()),
            doc_types=doc_types,
        )
        for key, value in download_stats.items():
            stats[key] += value

        if retry_failed:
            failed_rows = self.catalog.list_failed_documents(start_date=start_date, end_date=end_date, limit=200)
            self.logger.debug(
                "Retry-failed phase start rows=%s range=%s..%s",
                len(failed_rows),
                start_date.isoformat(),
                end_date.isoformat(),
            )
            for row in failed_rows:
                source_type = str(row["source_type"])
                if source_type == SourceType.URL_GUESS.value:
                    stats["retry_failed_skipped_url_guess"] += 1
                    continue
                stats["retry_failed_attempted"] += 1
                if await self._retry_document_row(row):
                    stats["retry_failed_recovered"] += 1
            self.logger.debug(
                "Retry-failed phase done attempted=%s recovered=%s skipped_url_guess=%s",
                stats["retry_failed_attempted"],
                stats["retry_failed_recovered"],
                stats["retry_failed_skipped_url_guess"],
            )
        else:
            self.logger.debug("Retry-failed phase skipped by configuration")

        return stats

    async def _discover_with_id_scan(
        self,
        scope_key: str,
        start_id: int,
        end_id: int,
        coarse_step: int,
        coarse_offsets: int,
        window: int,
        coarse_only: bool,
        phase_key: str,
        done_key: str,
        coarse_next_key: str,
        coarse_offsets_state_key: str,
        coarse_hits_key: str,
        fine_windows_key: str,
        fine_window_index_key: str,
        fine_next_id_key: str,
    ) -> dict[str, Any]:
        stats: dict[str, Any] = {"new_urls": 0, "probed": 0, "suggested_windows": []}
        phase = self.catalog.get_state(phase_key) or "coarse"
        legacy_coarse_next_id = int(self.catalog.get_state(coarse_next_key) or str(start_id))
        coarse_hits = self._load_int_state_list(coarse_hits_key)
        fine_windows = self._load_window_state_list(fine_windows_key)
        fine_window_index = int(self.catalog.get_state(fine_window_index_key) or "0")
        fine_next_id = int(self.catalog.get_state(fine_next_id_key) or "0")
        full_coarse_coverage = self._has_full_coarse_coverage(
            coarse_step=coarse_step,
            coarse_offsets=coarse_offsets,
        )

        if phase == "coarse":
            coarse_offsets_values = self._build_coarse_offsets(
                coarse_step=coarse_step,
                coarse_offsets=coarse_offsets,
            )
            coarse_state = self._load_coarse_offsets_state(
                key=coarse_offsets_state_key,
                start_id=start_id,
                end_id=end_id,
                coarse_step=coarse_step,
                coarse_offsets_values=coarse_offsets_values,
                legacy_next_id=legacy_coarse_next_id,
            )
            active_offsets = coarse_state["offsets"]
            next_ids = coarse_state["next_ids"]
            concurrency_limit = max(1, self.options.max_concurrency)
            offset_count = len(active_offsets)
            scan_cursor = 0

            while True:
                if offset_count == 0:
                    break

                batch: list[tuple[int, int]] = []
                next_ids_candidate = list(next_ids)
                while len(batch) < concurrency_limit:
                    selected = False
                    for shift in range(offset_count):
                        idx = (scan_cursor + shift) % offset_count
                        current_id = int(next_ids_candidate[idx])
                        if current_id > end_id:
                            continue
                        batch.append((idx, current_id))
                        next_ids_candidate[idx] = current_id + coarse_step
                        scan_cursor = (idx + 1) % offset_count
                        selected = True
                        break
                    if not selected:
                        break

                if not batch:
                    break

                probe_results = await asyncio.gather(
                    *[self._probe_id_candidate(current_id) for _, current_id in batch]
                )
                previous_probed = stats["probed"]
                stats["probed"] += len(batch)
                next_ids = next_ids_candidate
                self.catalog.set_state(
                    coarse_offsets_state_key,
                    json.dumps(
                        {
                            "step": coarse_step,
                            "offsets": active_offsets,
                            "next_ids": next_ids,
                        }
                    ),
                )
                self.catalog.set_state(coarse_next_key, str(min(next_ids)))

                for (_idx, current_id), probe in zip(batch, probe_results):
                    if probe["is_symbol"]:
                        coarse_hits.append(current_id)
                    if probe["is_target"] and probe["article_url"]:
                        inserted = self.catalog.enqueue_article_urls(
                            scope_key=scope_key,
                            source_type=SourceType.ID_SCAN,
                            urls=[str(probe["article_url"])],
                        )
                        stats["new_urls"] += inserted

                self.catalog.set_state(coarse_hits_key, json.dumps(sorted(set(coarse_hits))))
                if self._should_emit_progress(previous_probed, stats["probed"], interval=50):
                    self.logger.debug(
                        "Discovery[id_scan] coarse progress probed=%s offsets=%s next_id_min=%s symbol_hits=%s new_urls=%s",
                        stats["probed"],
                        len(active_offsets),
                        min(next_ids),
                        len(set(coarse_hits)),
                        stats["new_urls"],
                    )

            fine_windows = self._build_id_scan_windows(
                coarse_hits=sorted(set(coarse_hits)),
                start_id=start_id,
                end_id=end_id,
                window=window,
            )
            self.catalog.set_state(fine_windows_key, json.dumps(fine_windows))
            self.catalog.set_state(fine_window_index_key, "0")
            self.catalog.set_state(fine_next_id_key, str(fine_windows[0][0]) if fine_windows else "0")
            stats["suggested_windows"] = [[left, right] for left, right in fine_windows]
            phase = "done" if full_coarse_coverage else "fine"
            self.catalog.set_state(phase_key, phase)
            self.logger.info(
                "Discovery[id_scan] coarse done scope=%s symbol_hits=%s fine_windows=%s",
                scope_key,
                len(set(coarse_hits)),
                len(fine_windows),
            )
            if full_coarse_coverage:
                self.logger.info(
                    "Discovery[id_scan] fine skipped because coarse already covers all IDs for step=%s offsets=%s",
                    coarse_step,
                    coarse_offsets,
                )
                self.catalog.set_state(fine_window_index_key, str(len(fine_windows)))
                self.catalog.set_state(fine_next_id_key, "0")
            if coarse_only:
                self.logger.info(
                    "Discovery[id_scan] coarse-only stop scope=%s suggested_windows=%s",
                    scope_key,
                    len(fine_windows),
                )
                return stats

        if coarse_only:
            if not stats["suggested_windows"] and fine_windows:
                stats["suggested_windows"] = [[left, right] for left, right in fine_windows]
            self.logger.info(
                "Discovery[id_scan] coarse-only resume stop scope=%s phase=%s suggested_windows=%s",
                scope_key,
                phase,
                len(stats["suggested_windows"]),
            )
            return stats

        if phase == "fine" and full_coarse_coverage:
            self.logger.info(
                "Discovery[id_scan] fine resume skipped because coarse already covers all IDs for step=%s offsets=%s",
                coarse_step,
                coarse_offsets,
            )
            phase = "done"
            self.catalog.set_state(phase_key, phase)
            self.catalog.set_state(fine_window_index_key, str(len(fine_windows)))
            self.catalog.set_state(fine_next_id_key, "0")

        if phase == "fine":
            while fine_window_index < len(fine_windows):
                window_start, window_end = fine_windows[fine_window_index]
                current_id = fine_next_id if fine_next_id >= window_start else window_start
                concurrency_limit = max(1, self.options.max_concurrency)
                self.logger.debug(
                    "Discovery[id_scan] fine window=%s/%s range=%s-%s start_at=%s",
                    fine_window_index + 1,
                    len(fine_windows),
                    window_start,
                    window_end,
                    current_id,
                )
                while current_id <= window_end:
                    batch_end = min(window_end, current_id + concurrency_limit - 1)
                    batch_ids = list(range(current_id, batch_end + 1))
                    probe_results = await asyncio.gather(
                        *[self._probe_id_candidate(article_id) for article_id in batch_ids]
                    )
                    previous_probed = stats["probed"]
                    stats["probed"] += len(batch_ids)
                    for article_id, probe in zip(batch_ids, probe_results):
                        if probe["is_target"] and probe["article_url"]:
                            inserted = self.catalog.enqueue_article_urls(
                                scope_key=scope_key,
                                source_type=SourceType.ID_SCAN,
                                urls=[str(probe["article_url"])],
                            )
                            stats["new_urls"] += inserted
                    current_id = batch_end + 1
                    fine_next_id = current_id
                    self.catalog.set_state(fine_next_id_key, str(fine_next_id))
                    if self._should_emit_progress(previous_probed, stats["probed"], interval=200):
                        self.logger.debug(
                            "Discovery[id_scan] fine progress probed=%s current_window=%s/%s next_id=%s new_urls=%s",
                            stats["probed"],
                            fine_window_index + 1,
                            len(fine_windows),
                            fine_next_id,
                            stats["new_urls"],
                        )

                fine_window_index += 1
                self.catalog.set_state(fine_window_index_key, str(fine_window_index))
                fine_next_id = (
                    fine_windows[fine_window_index][0]
                    if fine_window_index < len(fine_windows)
                    else 0
                )
                self.catalog.set_state(fine_next_id_key, str(fine_next_id))

            phase = "done"
            self.catalog.set_state(phase_key, phase)

        if phase == "done":
            self.catalog.set_state(done_key, "1")
            self.logger.info(
                "Discovery[id_scan] done scope=%s id_range=%s:%s probed=%s new_urls=%s",
                scope_key,
                start_id,
                end_id,
                stats["probed"],
                stats["new_urls"],
            )

        return stats

    async def _probe_id_candidate(self, article_id: int) -> dict[str, Any]:
        probe_url = build_symbol_probe_url(article_id=article_id, symbol=self.options.symbol)
        html_text = await self._fetch_text(probe_url, "id_scan_probe")
        if not html_text:
            return {"is_symbol": False, "is_target": False, "article_url": None}

        detail = parse_article_detail(
            html_text=html_text,
            url=probe_url,
            source_type=SourceType.ID_SCAN,
        )

        symbol_signal_parts = [detail.title or "", detail.normalized_title or ""]
        if detail.url and detail.url != probe_url:
            symbol_signal_parts.append(detail.url)
        symbol_signal = normalize_text(" ".join(symbol_signal_parts))
        is_symbol = self.options.symbol.lower() in symbol_signal

        is_target = classify_doc_type(detail.title, detail.url) is not None
        if not is_target:
            for pdf_url in detail.pdf_urls:
                if classify_doc_type(detail.title, detail.url, pdf_url) is not None:
                    is_target = True
                    break
        return {
            "is_symbol": is_symbol,
            "is_target": is_target,
            "article_url": detail.url if is_target else None,
        }

    @staticmethod
    def _build_id_scan_windows(
        coarse_hits: list[int],
        start_id: int,
        end_id: int,
        window: int,
    ) -> list[tuple[int, int]]:
        if not coarse_hits:
            return []

        intervals: list[tuple[int, int]] = []
        for hit in coarse_hits:
            left = max(start_id, hit - window)
            right = min(end_id, hit + window)
            intervals.append((left, right))
        intervals.sort(key=lambda item: item[0])

        merged: list[tuple[int, int]] = []
        for left, right in intervals:
            if not merged:
                merged.append((left, right))
                continue
            prev_left, prev_right = merged[-1]
            if left <= prev_right + 1:
                merged[-1] = (prev_left, max(prev_right, right))
                continue
            merged.append((left, right))
        return merged

    @staticmethod
    def _build_coarse_offsets(coarse_step: int, coarse_offsets: int) -> list[int]:
        max_offsets = max(1, min(coarse_step, coarse_offsets))
        if max_offsets == 1:
            return [0]
        offsets = {int(i * coarse_step / max_offsets) for i in range(max_offsets)}
        return sorted(offsets)

    @staticmethod
    def _has_full_coarse_coverage(coarse_step: int, coarse_offsets: int) -> bool:
        if coarse_step < 1 or coarse_offsets < 1:
            return False
        return len(CafefScraperRunner._build_coarse_offsets(coarse_step, coarse_offsets)) == coarse_step

    @staticmethod
    def _should_emit_progress(previous: int, current: int, interval: int) -> bool:
        if previous <= 0 < current:
            return True
        if interval <= 0:
            return False
        return (previous // interval) < (current // interval)

    def _ensure_id_scan_resume_config(
        self,
        *,
        key: str,
        coarse_step: int,
        coarse_offsets: int,
        window: int,
    ) -> None:
        expected = {
            "coarse_step": coarse_step,
            "coarse_offsets": coarse_offsets,
            "window": window,
        }
        raw = self.catalog.get_state(key)
        if not raw:
            self.catalog.set_state(key, json.dumps(expected))
            return

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            parsed = {}

        try:
            existing = {
                "coarse_step": int(parsed.get("coarse_step")),
                "coarse_offsets": int(parsed.get("coarse_offsets")),
                "window": int(parsed.get("window")),
            }
        except (TypeError, ValueError):
            existing = {}

        if existing != expected:
            raise ValueError(
                "ID-scan resume parameters changed for this scope. "
                "Re-run with --reset-discovery-in-resume to restart ID-scan checkpoints."
            )
    def _load_coarse_offsets_state(
        self,
        key: str,
        start_id: int,
        end_id: int,
        coarse_step: int,
        coarse_offsets_values: list[int],
        legacy_next_id: int | None = None,
    ) -> dict[str, list[int]]:
        raw = self.catalog.get_state(key)
        expected_offsets = [int(v) for v in coarse_offsets_values]
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                step = int(parsed.get("step") or 0)
                offsets = parsed.get("offsets")
                next_ids = parsed.get("next_ids")
                if (
                    step == coarse_step
                    and isinstance(offsets, list)
                    and isinstance(next_ids, list)
                    and len(offsets) == len(next_ids) == len(expected_offsets)
                    and [int(v) for v in offsets] == expected_offsets
                ):
                    try:
                        return {
                            "offsets": [int(v) for v in offsets],
                            "next_ids": [int(v) for v in next_ids],
                        }
                    except (TypeError, ValueError):
                        pass

        initialized_next_ids = [start_id + offset for offset in expected_offsets]
        if legacy_next_id is not None and len(expected_offsets) == 1:
            initialized_next_ids = [legacy_next_id]
        normalized_next_ids = [value if value <= end_id else end_id + 1 for value in initialized_next_ids]
        state = {
            "step": coarse_step,
            "offsets": expected_offsets,
            "next_ids": normalized_next_ids,
        }
        self.catalog.set_state(key, json.dumps(state))
        return {"offsets": expected_offsets, "next_ids": normalized_next_ids}

    def _load_int_state_list(self, key: str) -> list[int]:
        raw = self.catalog.get_state(key)
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        values: list[int] = []
        for item in parsed:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                continue
        return values

    def _load_window_state_list(self, key: str) -> list[tuple[int, int]]:
        raw = self.catalog.get_state(key)
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        windows: list[tuple[int, int]] = []
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                left = int(item[0])
                right = int(item[1])
            except (TypeError, ValueError):
                continue
            if left > right:
                continue
            windows.append((left, right))
        return windows

    async def _process_article_queue(
        self,
        queue_rows: list[Any],
        start_date: date,
        end_date: date,
        doc_types: set[DocType],
        discovery_strategy: str,
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(max(1, self.options.max_concurrency))
        results_lock = asyncio.Lock()
        collected: list[tuple[int, DocumentCandidate]] = []
        counters = {"articles_fetched": 0, "article_fetch_failed": 0, "article_filtered_out": 0}
        total_queue = len(queue_rows)
        progress = {"processed": 0, "candidates": 0}

        self.logger.debug(
            "Article fetch phase start total=%s concurrency=%s",
            total_queue,
            max(1, self.options.max_concurrency),
        )

        async def _worker(row: Any) -> None:
            queue_id = int(row["id"])
            url = str(row["url"])
            source_type = SourceType(str(row["source_type"]))
            if (
                discovery_strategy == "balanced"
                and source_type != SourceType.ID_SCAN
                and not likely_target_article_url(url)
            ):
                self.catalog.mark_article_queue_filtered(
                    queue_id=queue_id,
                    reason="Filtered by balanced discovery strategy",
                )
                async with results_lock:
                    counters["article_filtered_out"] += 1
                    progress["processed"] += 1
                    self._debug_log_article_progress(
                        processed=progress["processed"],
                        total=total_queue,
                        fetched=counters["articles_fetched"],
                        failed=counters["article_fetch_failed"],
                        filtered=counters["article_filtered_out"],
                        candidates=progress["candidates"],
                    )
                return

            async with semaphore:
                html_text = await self._fetch_text(url, "article_page")
                if not html_text:
                    self.catalog.mark_article_queue_failed(queue_id=queue_id, error="Article fetch failed")
                    async with results_lock:
                        counters["article_fetch_failed"] += 1
                        progress["processed"] += 1
                        self._debug_log_article_progress(
                            processed=progress["processed"],
                            total=total_queue,
                            fetched=counters["articles_fetched"],
                            failed=counters["article_fetch_failed"],
                            filtered=counters["article_filtered_out"],
                            candidates=progress["candidates"],
                        )
                    return
                detail = parse_article_detail(html_text, url=url, source_type=source_type)
                self.catalog.upsert_article(
                    article_id=detail.article_id,
                    url=detail.url,
                    published_at=detail.published_at,
                    title=detail.title,
                    normalized_title=detail.normalized_title,
                    source_type=detail.source_type,
                )
                candidates: list[tuple[int, DocumentCandidate]] = []
                for pdf_url in detail.pdf_urls:
                    doc_type = classify_doc_type(detail.title, detail.url, pdf_url)
                    if doc_type is None or doc_type not in doc_types:
                        continue
                    event_date, derived = extract_event_date(
                        detail.title,
                        detail.url,
                        pdf_url,
                        published_at=detail.published_at,
                    )
                    if event_date is None or event_date < start_date or event_date > end_date:
                        continue
                    article_id = detail.article_id or self._synthetic_article_id(detail.url)
                    candidate = DocumentCandidate(
                        article_id=article_id,
                        article_url=detail.url,
                        source_type=source_type,
                        doc_type=doc_type,
                        event_date=event_date,
                        pdf_url=pdf_url,
                        slug=slugify(detail.title or Path(pdf_url).stem),
                        derived_from_published=derived,
                    )
                    doc_id = self.catalog.upsert_document(
                        article_id=candidate.article_id,
                        article_url=candidate.article_url,
                        source_type=candidate.source_type,
                        doc_type=candidate.doc_type,
                        event_date=candidate.event_date,
                        pdf_url=candidate.pdf_url,
                        derived_from_published=candidate.derived_from_published,
                    )
                    candidates.append((doc_id, candidate))
                self.catalog.mark_article_queue_fetched(queue_id=queue_id)
                async with results_lock:
                    counters["articles_fetched"] += 1
                    collected.extend(candidates)
                    progress["processed"] += 1
                    progress["candidates"] += len(candidates)
                    self._debug_log_article_progress(
                        processed=progress["processed"],
                        total=total_queue,
                        fetched=counters["articles_fetched"],
                        failed=counters["article_fetch_failed"],
                        filtered=counters["article_filtered_out"],
                        candidates=progress["candidates"],
                    )

        await asyncio.gather(*[asyncio.create_task(_worker(row)) for row in queue_rows])
        self.logger.debug(
            "Article fetch phase done total=%s fetched=%s failed=%s filtered=%s candidates=%s",
            total_queue,
            counters["articles_fetched"],
            counters["article_fetch_failed"],
            counters["article_filtered_out"],
            progress["candidates"],
        )
        return {"candidates": collected, **counters}

    async def _process_document_downloads(
        self,
        records: list[tuple[int, DocumentCandidate]],
        doc_types: set[DocType],
    ) -> dict[str, int]:
        stats = {
            "documents_downloaded": 0,
            "documents_failed": 0,
            "documents_skipped_existing": 0,
            "documents_skipped_duplicate": 0,
        }
        semaphore = asyncio.Semaphore(max(1, self.options.max_concurrency))
        lock = asyncio.Lock()
        failure_reasons: dict[str, int] = {}
        total_records = sum(1 for _, candidate in records if candidate.doc_type in doc_types)
        progress = {"processed": 0}

        self.logger.debug(
            "Document download phase start total=%s concurrency=%s dry_run=%s",
            total_records,
            max(1, self.options.max_concurrency),
            self.options.dry_run,
        )

        async def _worker(item: tuple[int, DocumentCandidate]) -> None:
            doc_id, candidate = item
            if candidate.doc_type not in doc_types:
                return
            existing = self.catalog.get_existing_found_document(
                doc_type=candidate.doc_type,
                event_date=candidate.event_date,
                pdf_url=candidate.pdf_url,
            )
            if existing and existing["local_path"] and Path(str(existing["local_path"])).exists():
                self.catalog.set_coverage_status(
                    target_date=candidate.event_date,
                    doc_type=candidate.doc_type,
                    status=CoverageStatus.FOUND,
                )
                async with lock:
                    stats["documents_skipped_existing"] += 1
                    progress["processed"] += 1
                    self._debug_log_download_progress(
                        processed=progress["processed"],
                        total=total_records,
                        stats=stats,
                    )
                return

            if self.options.dry_run:
                self.catalog.mark_document_discovered(doc_id)
                self.catalog.set_coverage_status(
                    target_date=candidate.event_date,
                    doc_type=candidate.doc_type,
                    status=CoverageStatus.FOUND,
                    note="dry-run discovered without download",
                )
                async with lock:
                    stats["documents_downloaded"] += 1
                    progress["processed"] += 1
                    self._debug_log_download_progress(
                        processed=progress["processed"],
                        total=total_records,
                        stats=stats,
                    )
                return

            target_path = self._build_pdf_path(candidate)
            self.logger.debug(
                "Downloading PDF doc_id=%s type=%s date=%s url=%s",
                doc_id,
                candidate.doc_type.value,
                candidate.event_date.isoformat(),
                candidate.pdf_url,
            )
            async with semaphore:
                result = await self._downloader.download_pdf(
                    url=candidate.pdf_url,
                    target_path=target_path,
                    stage="pdf_download",
                    on_failure=self._on_http_failure,
                )

            if result.success and result.local_path and result.sha256 and result.size_bytes is not None:
                duplicate = self.catalog.get_document_by_sha256(result.sha256)
                if duplicate and int(duplicate["id"]) != doc_id and duplicate["local_path"]:
                    if result.local_path.exists():
                        result.local_path.unlink(missing_ok=True)
                    self.catalog.mark_document_duplicate(
                        doc_id=doc_id,
                        local_path=str(duplicate["local_path"]),
                        sha256=result.sha256,
                        size_bytes=int(duplicate["size_bytes"] or result.size_bytes),
                    )
                    self.catalog.set_coverage_status(
                        target_date=candidate.event_date,
                        doc_type=candidate.doc_type,
                        status=CoverageStatus.FOUND,
                    )
                    async with lock:
                        stats["documents_skipped_duplicate"] += 1
                        progress["processed"] += 1
                        self._debug_log_download_progress(
                            processed=progress["processed"],
                            total=total_records,
                            stats=stats,
                        )
                    return

                self.catalog.mark_document_found(
                    doc_id=doc_id,
                    local_path=str(result.local_path),
                    sha256=result.sha256,
                    size_bytes=result.size_bytes,
                )
                self.catalog.set_coverage_status(
                    target_date=candidate.event_date,
                    doc_type=candidate.doc_type,
                    status=CoverageStatus.FOUND,
                )
                async with lock:
                    stats["documents_downloaded"] += 1
                    progress["processed"] += 1
                    self._debug_log_download_progress(
                        processed=progress["processed"],
                        total=total_records,
                        stats=stats,
                    )
                return

            self.catalog.mark_document_failed(doc_id=doc_id, error=result.error or "Download failed")
            reason_key = (result.error or "Download failed").strip()[:120]
            self.logger.warning(
                "Document download failed doc_id=%s type=%s date=%s status_code=%s error=%s url=%s",
                doc_id,
                candidate.doc_type.value,
                candidate.event_date.isoformat(),
                result.status_code,
                result.error or "Download failed",
                candidate.pdf_url,
            )
            current = self.catalog.get_coverage_status(candidate.event_date, candidate.doc_type)
            if current != CoverageStatus.FOUND:
                self.catalog.set_coverage_status(
                    target_date=candidate.event_date,
                    doc_type=candidate.doc_type,
                    status=CoverageStatus.FAILED,
                    note=result.error or "Download failed",
                )
            async with lock:
                stats["documents_failed"] += 1
                failure_reasons[reason_key] = failure_reasons.get(reason_key, 0) + 1
                progress["processed"] += 1
                self._debug_log_download_progress(
                    processed=progress["processed"],
                    total=total_records,
                    stats=stats,
                )

        await asyncio.gather(*[asyncio.create_task(_worker(item)) for item in records])
        if failure_reasons:
            failure_summary = "; ".join(
                [f"{reason} x{count}" for reason, count in sorted(failure_reasons.items(), key=lambda item: item[1], reverse=True)]
            )
            self.logger.debug(
                "Document download phase done total=%s downloaded=%s failed=%s skipped_existing=%s skipped_duplicate=%s failure_reasons=%s",
                total_records,
                stats["documents_downloaded"],
                stats["documents_failed"],
                stats["documents_skipped_existing"],
                stats["documents_skipped_duplicate"],
                failure_summary,
            )
        else:
            self.logger.debug(
                "Document download phase done total=%s downloaded=%s failed=%s skipped_existing=%s skipped_duplicate=%s",
                total_records,
                stats["documents_downloaded"],
                stats["documents_failed"],
                stats["documents_skipped_existing"],
                stats["documents_skipped_duplicate"],
            )
        return stats

    async def _retry_document_row(self, row: Any) -> bool:
        if self.options.dry_run:
            return False
        event_date = date.fromisoformat(str(row["event_date"]))
        doc_type = DocType(str(row["doc_type"]))
        candidate = DocumentCandidate(
            article_id=str(row["article_id"]),
            article_url=str(row["article_url"]),
            source_type=SourceType(str(row["source_type"])),
            doc_type=doc_type,
            event_date=event_date,
            pdf_url=str(row["pdf_url"]),
            slug=slugify(Path(str(row["pdf_url"])).stem),
            derived_from_published=bool(row["derived_from_published"]),
        )
        doc_id = int(row["id"])
        target_path = self._build_pdf_path(candidate)
        result = await self._downloader.download_pdf(
            url=candidate.pdf_url,
            target_path=target_path,
            stage="retry_pdf_download",
            on_failure=self._on_http_failure,
        )
        if result.success and result.local_path and result.sha256 and result.size_bytes is not None:
            self.catalog.mark_document_found(
                doc_id=doc_id,
                local_path=str(result.local_path),
                sha256=result.sha256,
                size_bytes=result.size_bytes,
            )
            self.catalog.set_coverage_status(
                target_date=event_date,
                doc_type=doc_type,
                status=CoverageStatus.FOUND,
            )
            return True
        self.catalog.mark_document_failed(doc_id=doc_id, error=result.error or "Retry failed")
        self.logger.warning(
            "Retry download failed doc_id=%s type=%s date=%s status_code=%s error=%s url=%s",
            doc_id,
            doc_type.value,
            event_date.isoformat(),
            result.status_code,
            result.error or "Retry failed",
            candidate.pdf_url,
        )
        return False

    async def _fetch_text(self, url: str, stage: str) -> str | None:
        max_retries: int | None = None
        timeout_seconds: float | None = None
        if stage == "id_scan_probe":
            max_retries = self.options.id_scan_probe_max_retries
            timeout_seconds = self.options.id_scan_probe_timeout_seconds
        return await self._downloader.fetch_text(
            url=url,
            stage=stage,
            on_failure=self._on_http_failure,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

    def _on_http_failure(
        self,
        url: str,
        stage: str,
        status_code: int | None,
        error: str | None,
        attempt_no: int,
    ) -> None:
        self.catalog.record_http_failure(
            run_id=self._current_run_id,
            url=url,
            stage=stage,
            status_code=status_code,
            error=error,
            attempt_no=attempt_no,
        )

    def _build_pdf_path(self, candidate: DocumentCandidate) -> Path:
        year = f"{candidate.event_date.year:04d}"
        month = f"{candidate.event_date.month:02d}"
        filename = (
            f"{candidate.event_date.isoformat()}__{candidate.article_id}__{candidate.slug}.pdf"
        )
        return (
            self.output_dir
            / "pdfs"
            / candidate.doc_type.value.lower()
            / year
            / month
            / filename
        )

    @staticmethod
    def _build_scope_key(mode: str, start_date: date, end_date: date) -> str:
        return f"{mode}:{start_date.isoformat()}:{end_date.isoformat()}"

    def _debug_log_article_progress(
        self,
        processed: int,
        total: int,
        fetched: int,
        failed: int,
        filtered: int,
        candidates: int,
    ) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        if not self._should_emit_debug_progress(processed, total, step=100):
            return
        self.logger.debug(
            "Article fetch progress processed=%s/%s fetched=%s failed=%s filtered=%s candidates=%s",
            processed,
            total,
            fetched,
            failed,
            filtered,
            candidates,
        )

    def _debug_log_download_progress(self, processed: int, total: int, stats: dict[str, int]) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        if not self._should_emit_debug_progress(processed, total, step=50):
            return
        self.logger.debug(
            "Download progress processed=%s/%s downloaded=%s failed=%s skipped_existing=%s skipped_duplicate=%s",
            processed,
            total,
            stats["documents_downloaded"],
            stats["documents_failed"],
            stats["documents_skipped_existing"],
            stats["documents_skipped_duplicate"],
        )

    @staticmethod
    def _should_emit_debug_progress(processed: int, total: int, step: int) -> bool:
        if total <= 0:
            return processed == 0
        return processed == 1 or processed == total or (processed % step == 0)

    @staticmethod
    def _synthetic_article_id(url: str) -> str:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return f"synthetic-{digest[:14]}"


def parse_doc_types(value: str) -> set[DocType]:
    normalized = value.strip().lower()
    if normalized == "both":
        return {DocType.BASKET_NOTICE, DocType.SWAP_END}
    if normalized == "basket":
        return {DocType.BASKET_NOTICE}
    if normalized in ("swap-end", "swap_end", "swap"):
        return {DocType.SWAP_END}
    raise ValueError(f"Unsupported doc-types value: {value}")


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{value}'. Expected YYYY-MM-DD.") from exc


def parse_id_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None

    match = re.fullmatch(r"(\d+)\s*:\s*(\d+)", normalized)
    if not match:
        raise ValueError(f"Invalid --id-range '{value}'. Expected START:END (e.g. 840000:950000).")

    start_id = int(match.group(1))
    end_id = int(match.group(2))
    if start_id <= 0 or end_id <= 0:
        raise ValueError("Invalid --id-range. START and END must be positive integers.")
    if start_id > end_id:
        raise ValueError("Invalid --id-range. START must be <= END.")
    return (start_id, end_id)


def today_vn() -> date:
    return datetime.now(tz=VN_TZ).date()


def format_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False)
