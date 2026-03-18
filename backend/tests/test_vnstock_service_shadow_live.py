from __future__ import annotations

import dataclasses
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


RUN_SERVICE_SHADOW = (
    os.getenv("RUN_VNSTOCK_SERVICE_SHADOW") == "1"
    or os.getenv("RUN_VNSTOCK_LIVE_DIFF") == "1"
)
pytestmark = pytest.mark.skipif(
    not RUN_SERVICE_SHADOW,
    reason="Set RUN_VNSTOCK_SERVICE_SHADOW=1 to run live backend service shadow checks.",
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


@pytest.fixture
def upstream_and_alt_vnstock_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, Any]:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    upstream = pytest.importorskip("vnstock_data", reason="vnstock_data must be installed for live differential checks.")
    alt = importlib.import_module("app.lib.vnstock_data_alt")
    return upstream, alt


@pytest.fixture
def alt_vnstock_data_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("app.lib.vnstock_data_alt")


def _date_window() -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    end = today - dt.timedelta(days=10)
    start = end - dt.timedelta(days=90)
    return start, end


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


def _flatten_columns(columns: Any) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flattened = []
        for item in columns:
            parts = [str(part) for part in item if part not in ("", None)]
            flattened.append("_".join(parts))
        return flattened
    return [str(col) for col in columns]


def _normalize_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    df = frame.copy()
    df.columns = _flatten_columns(df.columns)
    df = df.reset_index(drop=True)
    records = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): _normalize_scalar(value) for key, value in row.items()})
    return {
        "kind": "dataframe",
        "columns": list(df.columns),
        "rows": records,
    }


def _normalize_output(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_output(dataclasses.asdict(value))
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
        return {
            str(key): _normalize_output(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return _normalize_scalar(value)


def _reset_service_state() -> None:
    from app.services.vnstock_service.core import _fund_api_local, api_circuit_breaker
    from app.services.vnstock_service.stocks import StocksService

    api_circuit_breaker.reset()
    _fund_api_local.__dict__.clear()
    StocksService._industry_cache = {}
    StocksService._industry_cache_timestamp = 0


def _service_rate_limited() -> bool:
    from app.services.vnstock_service.core import api_circuit_breaker

    failure_count = getattr(api_circuit_breaker, "_failure_count", 0)
    return bool(failure_count)


def _make_funds_service(monkeypatch: pytest.MonkeyPatch):
    from app.services.vnstock_service.funds import FundsService

    service = FundsService()
    monkeypatch.setattr(service, "_get_fund_listing_records_from_db_sync", lambda fund_type=None: ([], None))
    monkeypatch.setattr(service, "_upsert_fund_listing_db_sync", lambda records: None)
    monkeypatch.setattr(service, "_get_fund_detail_cache_sync", lambda symbol, detail_type: ([], False))
    monkeypatch.setattr(service, "_upsert_fund_detail_cache_sync", lambda symbol, detail_type, data: None)
    return service


def _make_stocks_service():
    from app.services.vnstock_service.finance import FinanceService
    from app.services.vnstock_service.history import HistoryService
    from app.services.vnstock_service.stock_metadata import StockMetadataService
    from app.services.vnstock_service.stocks import StocksService

    metadata = StockMetadataService(finance_service=FinanceService())
    history = HistoryService()
    return StocksService(metadata=metadata, history=history)


def _run_service_call(
    monkeypatch: pytest.MonkeyPatch,
    package: Any,
    call: Callable[[pytest.MonkeyPatch], Any],
) -> dict[str, Any]:
    monkeypatch.setitem(sys.modules, "vnstock", package)
    _reset_service_state()
    try:
        return {
            "status": "ok",
            "value": _normalize_output(call(monkeypatch)),
            "rate_limited": _service_rate_limited(),
        }
    except Exception as exc:  # pragma: no cover - exercised on live failures
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
            "rate_limited": _service_rate_limited(),
        }


def _run_vnstock_data_service_call(
    monkeypatch: pytest.MonkeyPatch,
    package: Any,
    call: Callable[[pytest.MonkeyPatch], Any],
) -> dict[str, Any]:
    monkeypatch.setitem(sys.modules, "vnstock_data", package)
    _reset_service_state()
    try:
        return {
            "status": "ok",
            "value": _normalize_output(call(monkeypatch)),
            "rate_limited": False,
        }
    except Exception as exc:  # pragma: no cover - exercised on live failures
        message = str(exc)
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": message,
            "rate_limited": "429" in message or "rate limit" in message.lower(),
        }


SERVICE_LIVE_DIFF_CASES: list[tuple[str, Callable[[pytest.MonkeyPatch], Any]]] = [
    (
        "finance_income_statement",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.finance")
        .FinanceService()
        ._fetch_income_statement_sync("VCB", "en"),
    ),
    (
        "finance_balance_sheet",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.finance")
        .FinanceService()
        ._fetch_balance_sheet_sync("VCB", "en"),
    ),
    (
        "finance_cash_flow",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.finance")
        .FinanceService()
        ._fetch_cash_flow_sync("VCB", "en"),
    ),
    (
        "finance_ratios",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.finance")
        .FinanceService()
        ._fetch_financial_ratios_sync("VCB", "en"),
    ),
    (
        "company_overview",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.company")
        .CompanyService()
        ._fetch_company_overview_sync("VCB"),
    ),
    (
        "company_shareholders",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.company")
        .CompanyService()
        ._fetch_shareholders_sync("VCB"),
    ),
    (
        "company_officers",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.company")
        .CompanyService()
        ._fetch_officers_sync("VCB"),
    ),
    (
        "company_subsidiaries",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.company")
        .CompanyService()
        ._fetch_subsidiaries_sync("VCB"),
    ),
    (
        "indices_all_indices",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.indices")
        .IndicesService()
        ._fetch_all_indices_from_lib(),
    ),
    (
        "indices_index_value_vnindex",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.indices")
        .IndicesService()
        ._fetch_index_value_sync("VNINDEX"),
    ),
    (
        "stocks_industries",
        lambda monkeypatch: _make_stocks_service()._fetch_industries_sync(),
    ),
    (
        "stocks_industry_mapping",
        lambda monkeypatch: _make_stocks_service()._get_or_fetch_industry_mapping(),
    ),
    (
        "stocks_symbols_data",
        lambda monkeypatch: _make_stocks_service()._fetch_symbols_data(["VCB", "FPT"], 10),
    ),
    (
        "stocks_index_data_vn30",
        lambda monkeypatch: _make_stocks_service()._fetch_index_data("VN30", 10),
    ),
    (
        "funds_listing",
        lambda monkeypatch: _make_funds_service(monkeypatch)._fetch_fund_listing_sync(""),
    ),
    (
        "funds_nav_report",
        lambda monkeypatch: _make_funds_service(monkeypatch)._fetch_fund_nav_from_api_sync("SSISCA"),
    ),
    (
        "funds_top_holding",
        lambda monkeypatch: _make_funds_service(monkeypatch)._fetch_fund_top_holding_sync("SSISCA"),
    ),
    (
        "funds_industry_holding",
        lambda monkeypatch: _make_funds_service(monkeypatch)._fetch_fund_industry_holding_sync("SSISCA"),
    ),
    (
        "funds_asset_holding",
        lambda monkeypatch: _make_funds_service(monkeypatch)._fetch_fund_asset_holding_sync("SSISCA"),
    ),
    (
        "funds_benchmark_vnindex",
        lambda monkeypatch: _make_funds_service(monkeypatch)._fetch_benchmark_data_sync("VNINDEX", None),
    ),
    (
        "history_index_history_vnindex",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.history")
        .HistoryService()
        ._fetch_index_history_sync("VNINDEX", *_date_window()),
    ),
]

SERVICE_VNSTOCK_DATA_LIVE_DIFF_CASES: list[tuple[str, Callable[[pytest.MonkeyPatch], Any]]] = [
    (
        "history_foreign_trade_fetch_vcb",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.history")
        .HistoryService()
        ._fetch_foreign_trade_history("VCB", *_date_window()),
    ),
    (
        "history_prop_trade_fetch_vcb",
        lambda monkeypatch: importlib.import_module("app.services.vnstock_service.history")
        .HistoryService()
        ._fetch_prop_trade_history("VCB", *_date_window()),
    ),
]


@pytest.mark.parametrize("case_name,call", SERVICE_LIVE_DIFF_CASES, ids=[case[0] for case in SERVICE_LIVE_DIFF_CASES])
def test_backend_service_shadow_matches_upstream(
    case_name: str,
    call: Callable[[pytest.MonkeyPatch], Any],
    upstream_and_alt: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, alt = upstream_and_alt

    upstream_result = _run_service_call(monkeypatch, upstream, call)
    alt_result = _run_service_call(monkeypatch, alt, call)

    if upstream_result.get("rate_limited") or alt_result.get("rate_limited"):
        pytest.skip(f"Live rate limit encountered during {case_name}; skipping differential assertion.")

    assert alt_result == upstream_result, (
        f"Service shadow mismatch for {case_name}\n"
        f"upstream={json.dumps(upstream_result, ensure_ascii=False, sort_keys=True, default=str)[:4000]}\n"
        f"alt={json.dumps(alt_result, ensure_ascii=False, sort_keys=True, default=str)[:4000]}"
    )


@pytest.mark.parametrize(
    "case_name,call",
    SERVICE_VNSTOCK_DATA_LIVE_DIFF_CASES,
    ids=[case[0] for case in SERVICE_VNSTOCK_DATA_LIVE_DIFF_CASES],
)
def test_backend_history_aux_service_shadow_matches_upstream_vnstock_data(
    case_name: str,
    call: Callable[[pytest.MonkeyPatch], Any],
    upstream_and_alt_vnstock_data: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, alt = upstream_and_alt_vnstock_data

    upstream_result = _run_vnstock_data_service_call(monkeypatch, upstream, call)
    alt_result = _run_vnstock_data_service_call(monkeypatch, alt, call)

    if upstream_result.get("rate_limited") or alt_result.get("rate_limited"):
        pytest.skip(f"Live rate limit encountered during {case_name}; skipping differential assertion.")

    assert alt_result == upstream_result, (
        f"Service vnstock_data shadow mismatch for {case_name}\n"
        f"upstream={json.dumps(upstream_result, ensure_ascii=False, sort_keys=True, default=str)[:4000]}\n"
        f"alt={json.dumps(alt_result, ensure_ascii=False, sort_keys=True, default=str)[:4000]}"
    )


@pytest.mark.parametrize(
    "method_name,expected_columns",
    [
        ("_fetch_foreign_trade_history", {"time", "fr_buy_volume", "fr_buy_value", "fr_sell_volume", "fr_sell_value"}),
        ("_fetch_prop_trade_history", {"time", "prop_buy_volume", "prop_buy_value", "prop_sell_volume", "prop_sell_value"}),
    ],
    ids=["foreign_trade", "prop_trade"],
)
def test_backend_history_aux_live_with_alt_vnstock_data_only(
    method_name: str,
    expected_columns: set[str],
    alt_vnstock_data_only: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vnstock_data", alt_vnstock_data_only)

    service = importlib.import_module("app.services.vnstock_service.history").HistoryService()
    frame = getattr(service, method_name)("VCB", *_date_window())

    assert isinstance(frame, pd.DataFrame)
    if frame.empty:
        pytest.skip(f"{method_name} returned no live rows for the current date window")
    assert expected_columns.issubset(set(frame.columns))
