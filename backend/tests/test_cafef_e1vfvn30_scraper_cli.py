import pytest

from tools.cafef_e1vfvn30_scraper import build_parser


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--id-scan-coarse-step", "0"],
        ["--id-scan-coarse-step", "-1"],
        ["--id-scan-coarse-offsets", "0"],
        ["--id-scan-coarse-offsets", "-2"],
        ["--id-scan-window", "-1"],
        ["--id-scan-probe-max-retries", "0"],
        ["--id-scan-probe-timeout-seconds", "0.5"],
    ],
)
def test_backfill_rejects_invalid_id_scan_numeric_args(extra_args):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "/tmp/out", "backfill", *extra_args])


def test_backfill_accepts_step_one_and_window_zero():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "backfill",
            "--id-scan-coarse-step",
            "1",
            "--id-scan-coarse-offsets",
            "1",
            "--id-scan-window",
            "0",
        ]
    )
    assert args.id_scan_coarse_step == 1
    assert args.id_scan_coarse_offsets == 1
    assert args.id_scan_window == 0


def test_backfill_accepts_probe_stage_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--id-scan-probe-max-retries",
            "2",
            "--id-scan-probe-timeout-seconds",
            "8",
            "backfill",
        ]
    )
    assert args.id_scan_probe_max_retries == 2
    assert args.id_scan_probe_timeout_seconds == 8.0
