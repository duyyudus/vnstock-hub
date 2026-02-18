from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import timedelta
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.cafef_scraper.runner import (
    CafefScraperRunner,
    RunnerOptions,
    format_summary,
    parse_doc_types,
    parse_id_range,
    parse_iso_date,
    today_vn,
)


def _int_at_least(minimum: int, flag_name: str):
    def _parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{flag_name} must be an integer.") from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{flag_name} must be >= {minimum}.")
        return value

    return _parse


def _float_at_least(minimum: float, flag_name: str):
    def _parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{flag_name} must be a number.") from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{flag_name} must be >= {minimum}.")
        return value

    return _parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape CAFEF E1VFVN30 ETF documents "
            "(basket notice + swap-end) into local PDF storage + SQLite catalog."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Output folder for PDFs and catalog.sqlite")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--rate-limit-rps", type=float, default=1.5)
    parser.add_argument(
        "--adaptive-rate-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-adjust effective request rate on transient HTTP failures (429/5xx/request errors)",
    )
    parser.add_argument(
        "--adaptive-min-rps",
        type=float,
        default=0.5,
        help="Minimum floor for adaptive rate limiting",
    )
    parser.add_argument(
        "--adaptive-recovery-multiplier",
        type=float,
        default=1.05,
        help="Adaptive recovery multiplier after successful requests (lower = slower recovery)",
    )
    parser.add_argument(
        "--adaptive-cooldown-seconds",
        type=float,
        default=20.0,
        help="Cooldown hold time after repeated retryable failures",
    )
    parser.add_argument(
        "--adaptive-cooldown-streak",
        type=int,
        default=3,
        help="Number of consecutive retryable failures to trigger adaptive cooldown",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="HTTP max retry attempts for request errors and transient HTTP failures",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP request timeout seconds",
    )
    parser.add_argument(
        "--id-scan-probe-max-retries",
        type=_int_at_least(1, "--id-scan-probe-max-retries"),
        default=None,
        help="Optional override for max retries on id_scan_probe stage only (>=1)",
    )
    parser.add_argument(
        "--id-scan-probe-timeout-seconds",
        type=_float_at_least(1.0, "--id-scan-probe-timeout-seconds"),
        default=None,
        help="Optional override for timeout seconds on id_scan_probe stage only (>=1.0)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover and catalog without downloading PDFs")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser("backfill", help="Historical crawl")
    backfill.add_argument("--from-date", default="2014-01-01")
    backfill.add_argument("--to-date", default=today_vn().isoformat())
    backfill.add_argument("--doc-types", default="both", choices=("both", "basket", "swap-end"))
    backfill.add_argument(
        "--mode",
        default="full",
        choices=("full", "discover-sources", "discover-idscan", "consume-only"),
        help=(
            "full=discover+consume; discover-sources=only event/sitemap discovery; "
            "discover-idscan=only id-scan discovery; consume-only=fetch/download from queued discovered URLs"
        ),
    )
    backfill.add_argument("--max-event-pages", type=int, default=350)
    backfill.add_argument("--max-sitemaps", type=int, default=None)
    backfill.add_argument(
        "--event-feed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CAFEF event-feed discovery source",
    )
    backfill.add_argument(
        "--sitemap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CAFEF sitemap discovery source",
    )
    backfill.add_argument(
        "--discovery-strategy",
        default="balanced",
        choices=("balanced", "exhaustive"),
        help="balanced filters candidate article URLs before fetch; exhaustive fetches all discovered URLs",
    )
    backfill.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    backfill.add_argument(
        "--reset-discovery-in-resume",
        action="store_true",
        help="When using resume mode, reset discovery checkpoints for this range and re-crawl from scratch",
    )
    backfill.add_argument(
        "--retry-failed-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry previously failed document URLs within the run date range",
    )
    backfill.add_argument(
        "--id-range",
        default=None,
        help="Optional CAFEF article ID scan range START:END to recover legacy URLs missing from sitemap/feed",
    )
    backfill.add_argument(
        "--id-scan-coarse-step",
        type=_int_at_least(1, "--id-scan-coarse-step"),
        default=250,
        help="Coarse probe step size for --id-range scanning (>=1)",
    )
    backfill.add_argument(
        "--id-scan-coarse-offsets",
        type=_int_at_least(1, "--id-scan-coarse-offsets"),
        default=1,
        help="Number of evenly spaced modulo offsets sampled during coarse ID scan (>=1)",
    )
    backfill.add_argument(
        "--id-scan-window",
        type=_int_at_least(0, "--id-scan-window"),
        default=800,
        help="Fine-scan +/- window around coarse symbol hits (>=0, 0=exact hit IDs)",
    )
    backfill.add_argument(
        "--id-scan-coarse-only",
        action="store_true",
        help="Run only coarse ID scan and output suggested fine windows (skip fine scan/article fetch/download)",
    )

    incremental = subparsers.add_parser("incremental", help="Incremental crawl using lookback window")
    incremental.add_argument("--lookback-days", type=int, default=14)
    incremental.add_argument("--doc-types", default="both", choices=("both", "basket", "swap-end"))
    incremental.add_argument(
        "--mode",
        default="full",
        choices=("full", "discover-sources", "discover-idscan", "consume-only"),
        help=(
            "full=discover+consume; discover-sources=only event/sitemap discovery; "
            "discover-idscan=only id-scan discovery; consume-only=fetch/download from queued discovered URLs"
        ),
    )
    incremental.add_argument("--max-event-pages", type=int, default=120)
    incremental.add_argument("--max-sitemaps", type=int, default=350)
    incremental.add_argument(
        "--event-feed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CAFEF event-feed discovery source",
    )
    incremental.add_argument(
        "--sitemap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CAFEF sitemap discovery source",
    )
    incremental.add_argument(
        "--discovery-strategy",
        default="balanced",
        choices=("balanced", "exhaustive"),
        help="balanced filters candidate article URLs before fetch; exhaustive fetches all discovered URLs",
    )
    incremental.add_argument(
        "--reset-discovery-in-resume",
        action="store_true",
        help="Reset discovery checkpoints for this incremental scope and re-crawl from scratch",
    )
    incremental.add_argument(
        "--retry-failed-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry previously failed document URLs within the run date range",
    )
    incremental.add_argument(
        "--id-range",
        default=None,
        help="Optional CAFEF article ID scan range START:END to recover missing legacy URLs",
    )
    incremental.add_argument(
        "--id-scan-coarse-step",
        type=_int_at_least(1, "--id-scan-coarse-step"),
        default=250,
        help="Coarse probe step size for --id-range scanning (>=1)",
    )
    incremental.add_argument(
        "--id-scan-coarse-offsets",
        type=_int_at_least(1, "--id-scan-coarse-offsets"),
        default=1,
        help="Number of evenly spaced modulo offsets sampled during coarse ID scan (>=1)",
    )
    incremental.add_argument(
        "--id-scan-window",
        type=_int_at_least(0, "--id-scan-window"),
        default=800,
        help="Fine-scan +/- window around coarse symbol hits (>=0, 0=exact hit IDs)",
    )
    incremental.add_argument(
        "--id-scan-coarse-only",
        action="store_true",
        help="Run only coarse ID scan and output suggested fine windows (skip fine scan/article fetch/download)",
    )

    audit = subparsers.add_parser("audit", help="Coverage audit report")
    audit.add_argument("--from-date", default="2014-01-01")
    audit.add_argument("--to-date", default=today_vn().isoformat())
    audit.add_argument("--export-report", default=None)

    retry_failed = subparsers.add_parser("retry-failed", help="Retry failed docs from discovered article links")
    retry_failed.add_argument("--from-date", default=(today_vn() - timedelta(days=30)).isoformat())
    retry_failed.add_argument("--to-date", default=today_vn().isoformat())
    retry_failed.add_argument("--doc-types", default="both", choices=("both", "basket", "swap-end"))
    retry_failed.add_argument("--limit", type=int, default=300)

    return parser


async def _run_async(args: argparse.Namespace) -> int:
    logger = logging.getLogger("cafef_scraper")
    logger.setLevel(getattr(logging, args.log_level))
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.handlers.clear()
    logger.addHandler(handler)

    options = RunnerOptions(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        max_concurrency=max(1, args.max_concurrency),
        rate_limit_rps=max(0.1, args.rate_limit_rps),
        adaptive_rate_limit=bool(args.adaptive_rate_limit),
        adaptive_min_rps=max(0.1, args.adaptive_min_rps),
        adaptive_recovery_multiplier=max(1.01, args.adaptive_recovery_multiplier),
        adaptive_cooldown_seconds=max(0.0, args.adaptive_cooldown_seconds),
        adaptive_cooldown_streak=max(1, args.adaptive_cooldown_streak),
        dry_run=bool(args.dry_run),
        max_retries=max(1, args.max_retries),
        timeout_seconds=max(1.0, args.timeout_seconds),
        id_scan_probe_max_retries=args.id_scan_probe_max_retries,
        id_scan_probe_timeout_seconds=args.id_scan_probe_timeout_seconds,
    )
    runner = CafefScraperRunner(options=options, logger=logger)
    try:
        if args.command == "backfill":
            summary = await runner.run_backfill(
                start_date=parse_iso_date(args.from_date),
                end_date=parse_iso_date(args.to_date),
                doc_types=parse_doc_types(args.doc_types),
                resume=bool(args.resume),
                reset_discovery_in_resume=bool(args.reset_discovery_in_resume),
                mode=str(args.mode),
                discovery_strategy=str(args.discovery_strategy),
                max_event_pages=args.max_event_pages,
                max_sitemaps=args.max_sitemaps,
                event_feed_enabled=bool(args.event_feed),
                sitemap_enabled=bool(args.sitemap),
                id_range=parse_id_range(args.id_range),
                id_scan_coarse_step=args.id_scan_coarse_step,
                id_scan_coarse_offsets=args.id_scan_coarse_offsets,
                id_scan_window=args.id_scan_window,
                id_scan_coarse_only=bool(args.id_scan_coarse_only),
                retry_failed_enabled=bool(args.retry_failed_docs),
            )
        elif args.command == "incremental":
            summary = await runner.run_incremental(
                lookback_days=max(1, args.lookback_days),
                doc_types=parse_doc_types(args.doc_types),
                reset_discovery_in_resume=bool(args.reset_discovery_in_resume),
                mode=str(args.mode),
                discovery_strategy=str(args.discovery_strategy),
                max_event_pages=args.max_event_pages,
                max_sitemaps=args.max_sitemaps,
                event_feed_enabled=bool(args.event_feed),
                sitemap_enabled=bool(args.sitemap),
                id_range=parse_id_range(args.id_range),
                id_scan_coarse_step=args.id_scan_coarse_step,
                id_scan_coarse_offsets=args.id_scan_coarse_offsets,
                id_scan_window=args.id_scan_window,
                id_scan_coarse_only=bool(args.id_scan_coarse_only),
                retry_failed_enabled=bool(args.retry_failed_docs),
            )
        elif args.command == "retry-failed":
            summary = await runner.run_retry_failed(
                start_date=parse_iso_date(args.from_date),
                end_date=parse_iso_date(args.to_date),
                doc_types=parse_doc_types(args.doc_types),
                limit=max(1, args.limit),
            )
        elif args.command == "audit":
            export_path = Path(args.export_report).expanduser().resolve() if args.export_report else None
            summary = runner.run_audit(
                start_date=parse_iso_date(args.from_date),
                end_date=parse_iso_date(args.to_date),
                export_report=export_path,
            )
        else:
            raise ValueError(f"Unknown command {args.command}")

        print(format_summary(summary))
        return 0
    finally:
        await runner.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
