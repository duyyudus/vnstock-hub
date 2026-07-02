from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import os
import re
from typing import Any

import pandas as pd
import pytest


RUN_EXTENDED_LIVE_DIFF = (
    os.getenv("RUN_VNSTOCK_EXTENDED_LIVE_DIFF") == "1"
)


VNSTOCK_METHODS = {
    "Quote": ["history", "intraday", "price_depth"],
    "Company": ["affiliate", "events", "history", "news", "officers", "overview", "shareholders", "subsidiaries"],
    "Finance": ["balance_sheet", "cash_flow", "history", "income_statement", "ratio"],
    "Listing": [
        "all_bonds",
        "all_covered_warrant",
        "all_future_indices",
        "all_government_bonds",
        "all_symbols",
        "history",
        "industries_icb",
        "symbols_by_exchange",
        "symbols_by_group",
        "symbols_by_industries",
    ],
    "Trading": [
        "foreign_trade",
        "history",
        "insider_deal",
        "order_stats",
        "price_board",
        "price_history",
        "prop_trade",
        "side_stats",
        "trading_stats",
    ],
    "Fund": ["asset_holding", "filter", "industry_holding", "listing", "nav_report", "top_holding"],
}

VNSTOCK_DATA_METHODS = {
    "Quote": ["history", "intraday", "price_depth"],
    "Company": [
        "affiliate",
        "capital_history",
        "events",
        "history",
        "insider_trading",
        "news",
        "officers",
        "overview",
        "shareholders",
        "subsidiaries",
    ],
    "Finance": ["balance_sheet", "cash_flow", "history", "income_statement", "note", "ratio"],
    "Listing": [
        "all_bonds",
        "all_covered_warrant",
        "all_etf",
        "all_future_indices",
        "all_government_bonds",
        "all_indices",
        "all_symbols",
        "history",
        "indices_by_group",
        "industries_icb",
        "symbols_by_exchange",
        "symbols_by_group",
        "symbols_by_industries",
    ],
    "Trading": [
        "foreign_trade",
        "history",
        "insider_deal",
        "matched_by_price",
        "odd_lot",
        "order_stats",
        "price_board",
        "price_history",
        "prop_trade",
        "put_through",
        "side_stats",
        "trade_history",
        "trading_stats",
    ],
    "CommodityPrice": [
        "coke",
        "corn",
        "fertilizer_ure",
        "gas_natural",
        "gas_vn",
        "gold_global",
        "gold_vn",
        "history",
        "iron_ore",
        "oil_crude",
        "pork_china",
        "pork_north_vn",
        "soybean",
        "steel_d10",
        "steel_hrc",
        "sugar",
    ],
    "TopStock": ["deal", "foreign_buy", "foreign_sell", "gainer", "history", "loser", "value", "volume"],
    "Fund": ["asset_holding", "filter", "industry_holding", "listing", "nav_report", "top_holding"],
}


@pytest.fixture
def alt_packages() -> dict[str, Any]:
    return {
        "vnstock": importlib.import_module("app.lib.vnstock_alt"),
        "vnstock_data": importlib.import_module("app.lib.vnstock_data_alt"),
    }


def _date_window() -> tuple[str, str]:
    today = dt.date.today()
    end = today - dt.timedelta(days=7)
    start = end - dt.timedelta(days=60)
    return start.isoformat(), end.isoformat()


DATE_START, DATE_END = _date_window()


def _contract_for_scalar(value: Any) -> dict[str, Any]:
    return {"kind": "scalar", "type": type(value).__name__}


def _contract_for_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flattened: list[str] = []
        for item in df.columns:
            parts = [str(part) for part in item if part not in ("", None)]
            flattened.append("_".join(parts))
        df.columns = flattened
    else:
        df.columns = [str(col) for col in df.columns]
    return {
        "kind": "dataframe",
        "columns": list(df.columns),
        "empty": bool(df.empty),
    }


def _value_contract(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _value_contract(dataclasses.asdict(value))
    if isinstance(value, pd.DataFrame):
        return _contract_for_dataframe(value)
    if isinstance(value, pd.Series):
        return {
            "kind": "series",
            "name": str(value.name),
            "empty": bool(value.empty),
        }
    if isinstance(value, dict):
        nested = {str(key): _value_contract(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        return {
            "kind": "dict",
            "keys": list(nested.keys()),
            "value_kinds": {
                key: item.get("kind", "scalar") if isinstance(item, dict) else "scalar"
                for key, item in nested.items()
            },
        }
    if isinstance(value, list):
        first_contract = _value_contract(value[0]) if value else None
        return {
            "kind": "list",
            "empty": len(value) == 0,
            "item_kind": first_contract.get("kind", "scalar") if isinstance(first_contract, dict) else "scalar",
            "item_keys": first_contract.get("keys", []) if isinstance(first_contract, dict) else [],
            "item_columns": first_contract.get("columns", []) if isinstance(first_contract, dict) else [],
        }
    return _contract_for_scalar(value)


def _normalize_error_message(message: str) -> str:
    normalized = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", message)
    normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "DATE", normalized)
    normalized = re.sub(r"\b\d+\b", "N", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _capture_contract(callable_obj) -> dict[str, Any]:
    try:
        return {"status": "ok", "contract": _value_contract(callable_obj())}
    except BaseException as exc:  # pragma: no cover - exercised only on live failures
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": _normalize_error_message(str(exc)),
        }


def _is_rate_limit_result(result: dict[str, Any]) -> bool:
    message = str(result.get("message", "")).lower()
    return (
        "rate limit" in message
        or "giới hạn api" in message
        or "process terminated" in message
        or result.get("type") == "SystemExit"
    )


def _invoke_root_method(
    package: Any,
    class_name: str,
    ctor_kwargs: dict[str, Any] | None,
    method_name: str,
    method_args: tuple[Any, ...] = (),
    method_kwargs: dict[str, Any] | None = None,
) -> Any:
    ctor_kwargs = dict(ctor_kwargs or {})
    method_kwargs = dict(method_kwargs or {})
    instance = getattr(package, class_name)(**ctor_kwargs)
    method = getattr(instance, method_name)
    return method(*method_args, **method_kwargs)


FULL_SURFACE_CASES: list[tuple[str, str, str, dict[str, Any], str, tuple[Any, ...], dict[str, Any]]] = []


def _add_case(
    case_name: str,
    module_key: str,
    class_name: str,
    ctor_kwargs: dict[str, Any] | None,
    method_name: str,
    method_args: tuple[Any, ...] = (),
    method_kwargs: dict[str, Any] | None = None,
) -> None:
    FULL_SURFACE_CASES.append(
        (
            case_name,
            module_key,
            class_name,
            dict(ctor_kwargs or {}),
            method_name,
            tuple(method_args),
            dict(method_kwargs or {}),
        )
    )


# vnstock root public methods
_add_case("vnstock_quote_history", "vnstock", "Quote", {"source": "VCI", "symbol": "VCB", "show_log": False}, "history", method_kwargs={"start": DATE_START, "end": DATE_END, "interval": "1D"})
_add_case("vnstock_quote_intraday", "vnstock", "Quote", {"source": "KBS", "symbol": "VCB", "show_log": False}, "intraday", method_kwargs={"page_size": 50, "page": 1})
_add_case("vnstock_quote_price_depth", "vnstock", "Quote", {"source": "KBS", "symbol": "VCB", "show_log": False}, "price_depth")

for method_name in VNSTOCK_METHODS["Company"]:
    _add_case(
        f"vnstock_company_{method_name}",
        "vnstock",
        "Company",
        {"source": "VCI", "symbol": "VCB", "show_log": False},
        method_name,
    )

for method_name in VNSTOCK_METHODS["Finance"]:
    kwargs = {"period": "quarter", "lang": "en"} if method_name != "history" else {}
    _add_case(
        f"vnstock_finance_{method_name}",
        "vnstock",
        "Finance",
        {"source": "VCI", "symbol": "VCB", "show_log": False},
        method_name,
        method_kwargs=kwargs,
    )

listing_method_kwargs = {
    "symbols_by_group": {"group": "VN30"},
}
for method_name in VNSTOCK_METHODS["Listing"]:
    _add_case(
        f"vnstock_listing_{method_name}",
        "vnstock",
        "Listing",
        {"source": "VCI", "show_log": False},
        method_name,
        method_kwargs=listing_method_kwargs.get(method_name, {}),
    )

trading_method_kwargs = {
    "price_board": {"symbols_list": ["VCB", "FPT"]},
    "price_history": {"start": DATE_START, "end": DATE_END},
    "foreign_trade": {"start": DATE_START, "end": DATE_END},
    "prop_trade": {"start": DATE_START, "end": DATE_END},
    "insider_deal": {"start": DATE_START, "end": DATE_END},
    "order_stats": {"start": DATE_START, "end": DATE_END},
}
for method_name in VNSTOCK_METHODS["Trading"]:
    _add_case(
        f"vnstock_trading_{method_name}",
        "vnstock",
        "Trading",
        {"source": "VCI", "symbol": "VCB", "show_log": False},
        method_name,
        method_kwargs=trading_method_kwargs.get(method_name, {}),
    )

fund_method_kwargs = {"filter": {"symbol": "SSI"}}
for method_name in VNSTOCK_METHODS["Fund"]:
    _add_case(
        f"vnstock_fund_{method_name}",
        "vnstock",
        "Fund",
        {"random_agent": False},
        method_name,
        method_kwargs=fund_method_kwargs.get(method_name, {}),
    )

# vnstock_data root public methods
_add_case("vnstock_data_quote_history", "vnstock_data", "Quote", {"source": "VND", "symbol": "VCB", "show_log": False}, "history", method_kwargs={"start": DATE_START, "end": DATE_END, "interval": "1D"})
_add_case("vnstock_data_quote_intraday", "vnstock_data", "Quote", {"source": "KBS", "symbol": "VCB", "show_log": False}, "intraday", method_kwargs={"page_size": 50, "page": 1})
_add_case("vnstock_data_quote_price_depth", "vnstock_data", "Quote", {"source": "KBS", "symbol": "VCB", "show_log": False}, "price_depth")

company_sources = {
    "overview": "TVS",
}
for method_name in VNSTOCK_DATA_METHODS["Company"]:
    _add_case(
        f"vnstock_data_company_{method_name}",
        "vnstock_data",
        "Company",
        {"source": company_sources.get(method_name, "KBS"), "symbol": "VCB", "show_log": False},
        method_name,
    )

finance_case_config = {
    "balance_sheet": {"source": "KBS", "kwargs": {"period": "quarter", "lang": "en"}},
    "cash_flow": {"source": "VCI", "kwargs": {"period": "quarter", "lang": "en"}},
    "history": {"source": "VCI", "kwargs": {}},
    "income_statement": {"source": "MAS", "kwargs": {"period": "quarter", "lang": "en"}},
    "note": {"source": "VCI", "kwargs": {"period": "quarter", "lang": "en"}},
    "ratio": {"source": "MAS", "kwargs": {"period": "quarter", "lang": "en"}},
}
for method_name in VNSTOCK_DATA_METHODS["Finance"]:
    config = finance_case_config[method_name]
    _add_case(
        f"vnstock_data_finance_{method_name}",
        "vnstock_data",
        "Finance",
        {"source": config["source"], "symbol": "VCB", "show_log": False},
        method_name,
        method_kwargs=config["kwargs"],
    )

listing_ctor_by_method = {
    "all_indices": {},
    "indices_by_group": {},
    "all_symbols": {"source": "VND", "show_log": False},
}
listing_kwargs_by_method = {
    "symbols_by_group": {"group": "VN30"},
    "indices_by_group": {"group": "HOSE"},
}
for method_name in VNSTOCK_DATA_METHODS["Listing"]:
    _add_case(
        f"vnstock_data_listing_{method_name}",
        "vnstock_data",
        "Listing",
        listing_ctor_by_method.get(method_name, {"source": "KBS", "show_log": False}),
        method_name,
        method_kwargs=listing_kwargs_by_method.get(method_name, {}),
    )

trading_case_config = {
    "foreign_trade": {"source": "CAFEF", "kwargs": {"start": DATE_START, "end": DATE_END}},
    "history": {"source": "VCI", "kwargs": {}},
    "insider_deal": {"source": "CAFEF", "kwargs": {"start": DATE_START, "end": DATE_END}},
    "matched_by_price": {"source": "KBS", "kwargs": {}},
    "odd_lot": {"source": "KBS", "kwargs": {"symbols_list": ["VCB", "FPT"], "exchange": "HOSE"}},
    "order_stats": {"source": "CAFEF", "kwargs": {"start": DATE_START, "end": DATE_END}},
    "price_board": {"source": "VDS", "kwargs": {"symbols_list": ["VCB", "FPT"]}},
    "price_history": {"source": "CAFEF", "kwargs": {"start": DATE_START, "end": DATE_END}},
    "prop_trade": {"source": "CAFEF", "kwargs": {"start": DATE_START, "end": DATE_END}},
    "put_through": {"source": "KBS", "kwargs": {"exchange": "HOSE"}},
    "side_stats": {"source": "VCI", "kwargs": {}},
    "trade_history": {"source": "KBS", "kwargs": {}},
    "trading_stats": {"source": "VCI", "kwargs": {}},
}
for method_name in VNSTOCK_DATA_METHODS["Trading"]:
    config = trading_case_config[method_name]
    _add_case(
        f"vnstock_data_trading_{method_name}",
        "vnstock_data",
        "Trading",
        {"source": config["source"], "symbol": "VCB", "show_log": False},
        method_name,
        method_kwargs=config["kwargs"],
    )

for method_name in VNSTOCK_DATA_METHODS["CommodityPrice"]:
    _add_case(
        f"vnstock_data_commodity_{method_name}",
        "vnstock_data",
        "CommodityPrice",
        {"source": "SPL", "show_log": False},
        method_name,
        method_kwargs={"length": 8} if method_name != "history" else {},
    )

topstock_method_kwargs = {
    "gainer": {"index": "VNINDEX", "limit": 5},
    "loser": {"index": "VNINDEX", "limit": 5},
    "value": {"index": "VNINDEX", "limit": 5},
    "volume": {"index": "VNINDEX", "limit": 5},
    "deal": {"index": "VNINDEX", "limit": 5},
    "foreign_buy": {"limit": 5},
    "foreign_sell": {"limit": 5},
    "history": {},
}
for method_name in VNSTOCK_DATA_METHODS["TopStock"]:
    _add_case(
        f"vnstock_data_topstock_{method_name}",
        "vnstock_data",
        "TopStock",
        {"source": "VND"},
        method_name,
        method_kwargs=topstock_method_kwargs[method_name],
    )

fund_method_kwargs_data = {"filter": {"symbol": "SSI"}}
for method_name in VNSTOCK_DATA_METHODS["Fund"]:
    _add_case(
        f"vnstock_data_fund_{method_name}",
        "vnstock_data",
        "Fund",
        {"random_agent": False},
        method_name,
        method_kwargs=fund_method_kwargs_data.get(method_name, {}),
    )


def _covered_methods(module_key: str) -> dict[str, set[str]]:
    covered: dict[str, set[str]] = {}
    for _, case_module_key, class_name, _, method_name, _, _ in FULL_SURFACE_CASES:
        if case_module_key != module_key:
            continue
        covered.setdefault(class_name, set()).add(method_name)
    return covered


def test_full_surface_cases_cover_vnstock_root_public_methods() -> None:
    assert _covered_methods("vnstock") == {key: set(value) for key, value in VNSTOCK_METHODS.items()}


def test_full_surface_cases_cover_vnstock_data_root_public_methods() -> None:
    assert _covered_methods("vnstock_data") == {key: set(value) for key, value in VNSTOCK_DATA_METHODS.items()}


@pytest.mark.skipif(
    not RUN_EXTENDED_LIVE_DIFF,
    reason="Set RUN_VNSTOCK_EXTENDED_LIVE_DIFF=1 to run extended live full-surface checks.",
)
@pytest.mark.parametrize(
    "case_name,module_key,class_name,ctor_kwargs,method_name,method_args,method_kwargs",
    FULL_SURFACE_CASES,
    ids=[case[0] for case in FULL_SURFACE_CASES],
)
def test_extended_live_full_surface_contracts_are_callable(
    case_name: str,
    module_key: str,
    class_name: str,
    ctor_kwargs: dict[str, Any],
    method_name: str,
    method_args: tuple[Any, ...],
    method_kwargs: dict[str, Any],
    alt_packages: dict[str, Any],
) -> None:
    package = alt_packages[module_key]
    result = _capture_contract(
        lambda: _invoke_root_method(package, class_name, ctor_kwargs, method_name, method_args, method_kwargs)
    )

    if _is_rate_limit_result(result):
        pytest.skip(f"Live rate limit encountered during {case_name}; skipping full-surface assertion.")

    assert result["kind"] != "exception", f"{case_name}: {result}"


@pytest.mark.skipif(
    not RUN_EXTENDED_LIVE_DIFF,
    reason="Set RUN_VNSTOCK_EXTENDED_LIVE_DIFF=1 to run extended live full-surface checks.",
)
def test_vci_industry_fallback_sources_are_non_empty() -> None:
    vnstock_alt = importlib.import_module("app.lib.vnstock_alt")
    screener_module = importlib.import_module("app.lib.vnstock_data_alt.explorer.vci.screener")

    symbols = vnstock_alt.Listing(source="VCI", show_log=False).symbols_by_exchange()
    criteria = screener_module.Screener(show_log=False).get_criteria(to_df=False)

    assert isinstance(symbols, pd.DataFrame)
    assert not symbols.empty
    assert "icb_code2" in symbols.columns
    assert isinstance(criteria, list)
    assert any(item.get("name") == "sector" for item in criteria)
    assert any(item.get("name") == "sectorLv1" for item in criteria)
