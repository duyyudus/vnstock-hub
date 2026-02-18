import asyncio
import logging

import pytest

from tools.cafef_scraper.runner import CafefScraperRunner, RunnerOptions, parse_id_range


def test_parse_id_range_valid():
    assert parse_id_range("840000:950000") == (840000, 950000)


def test_parse_id_range_none_or_blank():
    assert parse_id_range(None) is None
    assert parse_id_range("   ") is None


@pytest.mark.parametrize(
    "value",
    [
        "abc:def",
        "950000:840000",
        "-1:100",
        "100",
        "100-200",
    ],
)
def test_parse_id_range_invalid(value: str):
    with pytest.raises(ValueError):
        parse_id_range(value)


def test_build_coarse_offsets_even_distribution():
    assert CafefScraperRunner._build_coarse_offsets(coarse_step=300, coarse_offsets=4) == [0, 75, 150, 225]


def test_build_coarse_offsets_caps_at_step():
    assert CafefScraperRunner._build_coarse_offsets(coarse_step=5, coarse_offsets=20) == [0, 1, 2, 3, 4]


def test_has_full_coarse_coverage_for_step_one():
    assert CafefScraperRunner._has_full_coarse_coverage(coarse_step=1, coarse_offsets=1) is True


def test_has_full_coarse_coverage_for_offsets_equal_step():
    assert CafefScraperRunner._has_full_coarse_coverage(coarse_step=5, coarse_offsets=5) is True


def test_has_full_coarse_coverage_false_when_offsets_below_step():
    assert CafefScraperRunner._has_full_coarse_coverage(coarse_step=5, coarse_offsets=4) is False


def test_build_id_scan_windows_supports_zero_window():
    assert CafefScraperRunner._build_id_scan_windows(
        coarse_hits=[100, 102],
        start_id=90,
        end_id=110,
        window=0,
    ) == [(100, 100), (102, 102)]


def test_resume_id_scan_config_mismatch_requires_reset(tmp_path):
    runner = CafefScraperRunner(
        options=RunnerOptions(output_dir=tmp_path / "out"),
        logger=logging.getLogger("test_cafef_scraper_runner"),
    )
    try:
        key = "discovery:backfill:2026-01-01:2026-01-02:id_scan:900000-901000:config"
        runner._ensure_id_scan_resume_config(
            key=key,
            coarse_step=1,
            coarse_offsets=1,
            window=0,
        )
        with pytest.raises(ValueError, match="--reset-discovery-in-resume"):
            runner._ensure_id_scan_resume_config(
                key=key,
                coarse_step=5,
                coarse_offsets=1,
                window=0,
            )
    finally:
        asyncio.run(runner.close())


def test_discover_with_id_scan_skips_fine_when_coarse_is_full_coverage(tmp_path):
    runner = CafefScraperRunner(
        options=RunnerOptions(output_dir=tmp_path / "out"),
        logger=logging.getLogger("test_cafef_scraper_runner"),
    )
    calls: list[int] = []

    async def _fake_probe(article_id: int):
        calls.append(article_id)
        return {"is_symbol": True, "is_target": False, "article_url": None}

    runner._probe_id_candidate = _fake_probe  # type: ignore[method-assign]

    try:
        stats = asyncio.run(
            runner._discover_with_id_scan(
                scope_key="backfill:2026-01-01:2026-01-01",
                start_id=10,
                end_id=12,
                coarse_step=1,
                coarse_offsets=1,
                window=0,
                coarse_only=False,
                phase_key="phase",
                done_key="done",
                coarse_next_key="coarse_next",
                coarse_offsets_state_key="coarse_offsets_state",
                coarse_hits_key="coarse_hits",
                fine_windows_key="fine_windows",
                fine_window_index_key="fine_window_index",
                fine_next_id_key="fine_next_id",
            )
        )
        assert stats["probed"] == 3
        assert calls == [10, 11, 12]
        assert runner.catalog.get_state("done") == "1"
    finally:
        asyncio.run(runner.close())


def test_discover_with_id_scan_uses_max_concurrency(tmp_path):
    runner = CafefScraperRunner(
        options=RunnerOptions(output_dir=tmp_path / "out", max_concurrency=2),
        logger=logging.getLogger("test_cafef_scraper_runner"),
    )
    calls: list[int] = []
    inflight = 0
    max_inflight = 0

    async def _fake_probe(article_id: int):
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        calls.append(article_id)
        await asyncio.sleep(0.01)
        inflight -= 1
        return {"is_symbol": True, "is_target": False, "article_url": None}

    runner._probe_id_candidate = _fake_probe  # type: ignore[method-assign]

    try:
        stats = asyncio.run(
            runner._discover_with_id_scan(
                scope_key="backfill:2026-01-01:2026-01-01",
                start_id=1,
                end_id=4,
                coarse_step=1,
                coarse_offsets=1,
                window=0,
                coarse_only=False,
                phase_key="phase_parallel",
                done_key="done_parallel",
                coarse_next_key="coarse_next_parallel",
                coarse_offsets_state_key="coarse_offsets_state_parallel",
                coarse_hits_key="coarse_hits_parallel",
                fine_windows_key="fine_windows_parallel",
                fine_window_index_key="fine_window_index_parallel",
                fine_next_id_key="fine_next_id_parallel",
            )
        )
        assert stats["probed"] == 4
        assert calls == [1, 2, 3, 4]
        assert max_inflight >= 2
    finally:
        asyncio.run(runner.close())


def test_fetch_text_uses_probe_stage_overrides(tmp_path):
    runner = CafefScraperRunner(
        options=RunnerOptions(
            output_dir=tmp_path / "out",
            id_scan_probe_max_retries=2,
            id_scan_probe_timeout_seconds=8.0,
        ),
        logger=logging.getLogger("test_cafef_scraper_runner"),
    )
    captured: list[tuple[str, int | None, float | None]] = []

    async def _fake_fetch_text(url: str, stage: str, on_failure=None, max_retries=None, timeout_seconds=None):
        captured.append((stage, max_retries, timeout_seconds))
        return "ok"

    runner._downloader.fetch_text = _fake_fetch_text  # type: ignore[method-assign]
    try:
        first = asyncio.run(runner._fetch_text("https://example.com/a", "id_scan_probe"))
        second = asyncio.run(runner._fetch_text("https://example.com/b", "article_page"))
        assert first == "ok"
        assert second == "ok"
        assert captured == [
            ("id_scan_probe", 2, 8.0),
            ("article_page", None, None),
        ]
    finally:
        asyncio.run(runner.close())
