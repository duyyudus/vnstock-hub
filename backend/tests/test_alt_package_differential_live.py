from __future__ import annotations

import datetime as dt
import importlib
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest


RUN_LIVE_DIFF = os.getenv("RUN_VNSTOCK_LIVE_DIFF") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_LIVE_DIFF,
    reason="Set RUN_VNSTOCK_LIVE_DIFF=1 to run live upstream-vs-alt differential checks.",
)


def _prepare_upstream_vnstock_data_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in list(sys.modules):
        if name == "vnstock_data" or name.startswith("vnstock_data."):
            sys.modules.pop(name, None)

    fake_home = tmp_path / "fake-home"
    (fake_home / ".vnstock").mkdir(parents=True, exist_ok=True)
    (fake_home / ".vnstock" / "user.json").write_text('{"user": true}')
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    stub = types.ModuleType("vnii")
    stub.lc_init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vnii", stub)


@pytest.fixture
def upstream_and_alt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, Any]:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    upstream = importlib.import_module("vnstock")
    alt = importlib.import_module("app.lib.vnstock_alt")
    return upstream, alt


def _date_window() -> tuple[str, str]:
    today = dt.date.today()
    end = today - dt.timedelta(days=5)
    start = end - dt.timedelta(days=45)
    return start.isoformat(), end.isoformat()


def _flatten_columns(columns: Any) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flattened = []
        for item in columns:
            parts = [str(part) for part in item if part not in ("", None)]
            flattened.append("_".join(parts))
        return flattened
    return [str(col) for col in columns]


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 8)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _normalize_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    df = frame.copy()
    df.columns = _flatten_columns(df.columns)
    df = df.reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        normalized = {str(key): _normalize_scalar(value) for key, value in row.items()}
        records.append(normalized)

    records.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
    return {
        "kind": "dataframe",
        "columns": list(df.columns),
        "rows": records,
    }


def _normalize_output(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _normalize_dataframe(value)
    if isinstance(value, pd.Series):
        return {
            "kind": "series",
            "name": str(value.name),
            "values": [_normalize_scalar(item) for item in value.tolist()],
        }
    if isinstance(value, tuple):
        return [_normalize_output(item) for item in value]
    if isinstance(value, list):
        return [_normalize_output(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_output(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    return _normalize_scalar(value)


def _capture(call: Callable[[Any], Any], package: Any) -> dict[str, Any]:
    try:
        return {"status": "ok", "value": _normalize_output(call(package))}
    except Exception as exc:  # pragma: no cover - exercised only on live failures
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
        }


LIVE_DIFF_CASES: list[tuple[str, Callable[[Any], Any]]] = [
    ("listing_all_indices", lambda pkg: pkg.Listing(source="VCI", show_log=False).all_indices()),
    ("listing_industries_icb", lambda pkg: pkg.Listing(source="VCI", show_log=False).industries_icb()),
    ("listing_symbols_by_industries", lambda pkg: pkg.Listing(source="VCI", show_log=False).symbols_by_industries()),
    ("listing_symbols_by_group_vn30", lambda pkg: pkg.Listing(source="VCI", show_log=False).symbols_by_group("VN30")),
    ("listing_all_symbols", lambda pkg: pkg.Listing(source="VCI", show_log=False).all_symbols()),
    ("trading_price_board", lambda pkg: pkg.Trading(source="VCI", show_log=False).price_board(["VCB", "FPT"])),
    (
        "quote_history_daily",
        lambda pkg: pkg.Vnstock().stock(symbol="VCB", source="VCI").quote.history(
            start=_date_window()[0],
            end=_date_window()[1],
            interval="1D",
        ),
    ),
    (
        "finance_income_statement",
        lambda pkg: pkg.Vnstock().stock(symbol="VCB", source="VCI").finance.income_statement(period="quarter", lang="en"),
    ),
    (
        "finance_balance_sheet",
        lambda pkg: pkg.Vnstock().stock(symbol="VCB", source="VCI").finance.balance_sheet(period="quarter", lang="en"),
    ),
    (
        "finance_cash_flow",
        lambda pkg: pkg.Vnstock().stock(symbol="VCB", source="VCI").finance.cash_flow(period="quarter", lang="en"),
    ),
    (
        "finance_ratio",
        lambda pkg: pkg.Vnstock().stock(symbol="VCB", source="VCI").finance.ratio(period="quarter", lang="en"),
    ),
    ("company_overview", lambda pkg: pkg.Company(symbol="VCB", source="VCI", show_log=False).overview()),
    ("company_shareholders", lambda pkg: pkg.Company(symbol="VCB", source="VCI", show_log=False).shareholders()),
    ("company_officers", lambda pkg: pkg.Company(symbol="VCB", source="VCI", show_log=False).officers()),
    ("company_subsidiaries", lambda pkg: pkg.Company(symbol="VCB", source="VCI", show_log=False).subsidiaries()),
    ("fund_listing", lambda pkg: pkg.Fund().listing()),
    ("fund_nav_report", lambda pkg: pkg.Fund().details.nav_report("SSISCA")),
    ("fund_top_holding", lambda pkg: pkg.Fund().details.top_holding("SSISCA")),
    ("fund_industry_holding", lambda pkg: pkg.Fund().details.industry_holding("SSISCA")),
    ("fund_asset_holding", lambda pkg: pkg.Fund().details.asset_holding("SSISCA")),
]


@pytest.mark.parametrize("case_name,call", LIVE_DIFF_CASES, ids=[case[0] for case in LIVE_DIFF_CASES])
def test_upstream_and_alt_match_for_backend_used_surface(
    case_name: str,
    call: Callable[[Any], Any],
    upstream_and_alt: tuple[Any, Any],
) -> None:
    upstream, alt = upstream_and_alt

    upstream_result = _capture(call, upstream)
    alt_result = _capture(call, alt)

    assert alt_result == upstream_result, (
        f"Differential mismatch for {case_name}\n"
        f"upstream={json.dumps(upstream_result, ensure_ascii=False, sort_keys=True, default=str)[:4000]}\n"
        f"alt={json.dumps(alt_result, ensure_ascii=False, sort_keys=True, default=str)[:4000]}"
    )
