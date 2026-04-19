from __future__ import annotations

import importlib
import inspect
import sys
import types
import ast
from enum import Enum
from pathlib import Path

import pandas as pd
import pytest


ALT_ROOT = Path(__file__).resolve().parents[1] / "app" / "lib"


@pytest.fixture(autouse=True)
def _set_mplconfig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))


def _import(module_name: str):
    return importlib.import_module(module_name)


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

    stub = types.ModuleType("vnii")
    stub.lc_init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vnii", stub)


def test_vnstock_alt_root_exports_smoke() -> None:
    module = _import("app.lib.vnstock_alt")

    for name in ["Vnstock", "Quote", "Company", "Finance", "Listing", "Trading", "Fund"]:
        assert hasattr(module, name)

    for omitted in ["register_user", "change_api_key", "check_status"]:
        assert not hasattr(module, omitted)


def test_vnstock_data_alt_root_exports_smoke() -> None:
    module = _import("app.lib.vnstock_data_alt")

    for name in [
        "Quote",
        "Company",
        "Finance",
        "Listing",
        "Trading",
        "CommodityPrice",
        "TopStock",
        "Fund",
        "Reference",
        "Market",
        "Insights",
        "Fundamental",
        "Macro",
        "Analytics",
        "show_api",
        "show_doc",
    ]:
        assert hasattr(module, name)


@pytest.mark.parametrize(
    "module_name",
    [
        "app.lib.vnstock_alt.explorer.vci",
        "app.lib.vnstock_alt.explorer.kbs",
        "app.lib.vnstock_alt.explorer.msn",
        "app.lib.vnstock_alt.explorer.fmarket",
        "app.lib.vnstock_data_alt.explorer.vci",
        "app.lib.vnstock_data_alt.explorer.kbs",
        "app.lib.vnstock_data_alt.explorer.mas",
        "app.lib.vnstock_data_alt.explorer.tvs",
        "app.lib.vnstock_data_alt.explorer.vnd",
        "app.lib.vnstock_data_alt.explorer.vds",
        "app.lib.vnstock_data_alt.explorer.cafef",
        "app.lib.vnstock_data_alt.explorer.spl",
        "app.lib.vnstock_data_alt.explorer.mbk",
        "app.lib.vnstock_data_alt.explorer.fmarket",
        "app.lib.vnstock_data_alt.ui.helper",
    ],
)
def test_representative_submodules_import(module_name: str) -> None:
    assert _import(module_name)


def test_viz_stubs_raise_clear_import_error() -> None:
    shared_viz = _import("app.lib._vnstock_shared.common.viz")
    alt_viz = _import("app.lib.vnstock_alt.common.viz")

    assert shared_viz.HAS_VNSTOCK_CHART is False
    assert shared_viz.HAS_VNSTOCK_EZCHART is False
    assert alt_viz.HAS_VNSTOCK_CHART is False
    assert alt_viz.HAS_VNSTOCK_EZCHART is False

    with pytest.raises(ImportError, match="Charting helpers are intentionally not bundled"):
        shared_viz.Chart(None)

    with pytest.raises(ImportError, match="Charting helpers are intentionally not bundled"):
        alt_viz.get_chart(None)


def test_upgrade_helpers_are_disabled_noops() -> None:
    shared_upgrade = _import("app.lib._vnstock_shared.core.utils.upgrade")
    alt_upgrade = _import("app.lib.vnstock_alt.core.utils.upgrade")

    assert shared_upgrade.detect_environment() == "Terminal"
    assert alt_upgrade.detect_environment() == "Terminal"
    assert shared_upgrade.update_notice() is None
    assert alt_upgrade.update_notice(verbose=True) is None
    assert shared_upgrade.show_full_notice() is None


def test_colab_helpers_are_reduced_to_local_path_shims() -> None:
    shared_ggcolab = _import("app.lib._vnstock_shared.core.config.ggcolab")
    alt_env = _import("app.lib.vnstock_alt.core.utils.env")

    expected = Path.home() / ".vnstock"

    assert shared_ggcolab.is_google_colab() is False
    assert shared_ggcolab.is_drive_mounted() is False
    assert shared_ggcolab.get_vnstock_directory() == expected
    assert shared_ggcolab.get_install_target() is None
    assert shared_ggcolab.get_install_command() == ""
    assert shared_ggcolab.setup_colab_drive() is False
    assert shared_ggcolab.migrate_vnstock_data_colab() is False

    assert alt_env.get_vnstock_directory() == expected
    assert alt_env.get_vnstock_path() == expected
    assert alt_env.is_colab() is False
    assert alt_env.setup_colab_drive() is False
    assert alt_env.get_colab_install_command() == ""
    assert alt_env.show_colab_instructions() is None
    assert alt_env.id_valid() is True


def test_alt_packages_are_self_contained() -> None:
    offenders: list[str] = []
    blocked_roots = {"vnstock", "vnstock_data", "vnai"}

    for package in ["vnstock_alt", "vnstock_data_alt"]:
        for path in (ALT_ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split(".", 1)[0]
                        if module_name in blocked_roots:
                            offenders.append(f"{path}:{node.lineno} import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module_name = node.module.split(".", 1)[0]
                    if module_name in blocked_roots:
                        offenders.append(f"{path}:{node.lineno} from {node.module} import ...")

    assert offenders == []


@pytest.mark.parametrize(
    ("left_expr", "right_expr"),
    [
        ("app.lib.vnstock_alt.Vnstock.__new__", "vnstock.Vnstock.__new__"),
        ("app.lib.vnstock_alt.Listing.__init__", "vnstock.Listing.__init__"),
        ("app.lib.vnstock_alt.Listing.all_symbols", "vnstock.Listing.all_symbols"),
        ("app.lib.vnstock_alt.Listing.symbols_by_group", "vnstock.Listing.symbols_by_group"),
        ("app.lib.vnstock_alt.Trading.price_board", "vnstock.Trading.price_board"),
        ("app.lib.vnstock_alt.Company.overview", "vnstock.Company.overview"),
        ("app.lib.vnstock_alt.Finance.income_statement", "vnstock.Finance.income_statement"),
        ("app.lib.vnstock_data_alt.Listing.__init__", "vnstock_data.Listing.__init__"),
        ("app.lib.vnstock_data_alt.Listing.all_indices", "vnstock_data.Listing.all_indices"),
        ("app.lib.vnstock_data_alt.Listing.indices_by_group", "vnstock_data.Listing.indices_by_group"),
        ("app.lib.vnstock_data_alt.Quote.history", "vnstock_data.Quote.history"),
        ("app.lib.vnstock_data_alt.show_api", "vnstock_data.show_api"),
        ("app.lib.vnstock_data_alt.show_doc", "vnstock_data.show_doc"),
    ],
)
def test_selected_signatures_match_upstream(
    left_expr: str,
    right_expr: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    namespace: dict[str, object] = {}
    exec(
        f"import app.lib.vnstock_alt, app.lib.vnstock_data_alt, vnstock, vnstock_data\nleft = {left_expr}\nright = {right_expr}",
        namespace,
    )
    assert inspect.signature(namespace["left"]) == inspect.signature(namespace["right"])


VNSTOCK_ROOT_METHODS = {
    "Vnstock": [],
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

VNSTOCK_DATA_ROOT_METHODS = {
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


def _public_methods(cls: type) -> list[str]:
    names: list[str] = []
    for name, member in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) or inspect.ismethoddescriptor(member):
            names.append(name)
    return names


@pytest.mark.parametrize("class_name,expected_methods", VNSTOCK_ROOT_METHODS.items())
def test_vnstock_alt_root_public_methods_match_upstream(
    class_name: str,
    expected_methods: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    upstream = _import("vnstock")
    alt = _import("app.lib.vnstock_alt")

    assert _public_methods(getattr(upstream, class_name)) == expected_methods
    alt_methods = _public_methods(getattr(alt, class_name))
    if class_name == "Company":
        assert set(expected_methods) <= set(alt_methods)
        assert {"ownership", "capital_history", "insider_trading"} <= set(alt_methods)
    else:
        assert alt_methods == expected_methods


@pytest.mark.parametrize("class_name,expected_methods", VNSTOCK_DATA_ROOT_METHODS.items())
def test_vnstock_data_alt_root_public_methods_match_upstream(
    class_name: str,
    expected_methods: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    upstream = _import("vnstock_data")
    alt = _import("app.lib.vnstock_data_alt")

    assert _public_methods(getattr(upstream, class_name)) == expected_methods
    alt_methods = _public_methods(getattr(alt, class_name))
    if class_name == "Company":
        assert set(expected_methods) <= set(alt_methods)
        assert {"ownership"} <= set(alt_methods)
    else:
        assert alt_methods == expected_methods


def test_vnstock_data_alt_index_group_enum_matches_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    upstream = _import("vnstock_data")
    alt = _import("app.lib.vnstock_data_alt")

    upstream_enum = getattr(upstream, "IndexGroup")
    alt_enum = getattr(alt, "IndexGroup")

    assert issubclass(upstream_enum, Enum)
    assert issubclass(alt_enum, Enum)
    assert [item.value for item in alt_enum] == [item.value for item in upstream_enum]


@pytest.mark.parametrize(
    ("left_expr", "right_expr"),
    [
        ("app.lib.vnstock_data_alt.Reference.__init__", "vnstock_data.Reference.__init__"),
        ("app.lib.vnstock_data_alt.Market.__init__", "vnstock_data.Market.__init__"),
        ("app.lib.vnstock_data_alt.Insights.__init__", "vnstock_data.Insights.__init__"),
        ("app.lib.vnstock_data_alt.Fundamental.__init__", "vnstock_data.Fundamental.__init__"),
        ("app.lib.vnstock_data_alt.Macro.__init__", "vnstock_data.Macro.__init__"),
        ("app.lib.vnstock_data_alt.Analytics.__init__", "vnstock_data.Analytics.__init__"),
    ],
)
def test_vnstock_data_alt_ui_entrypoint_signatures_match_upstream(
    left_expr: str,
    right_expr: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _prepare_upstream_vnstock_data_import(monkeypatch, tmp_path)
    namespace: dict[str, object] = {}
    exec(
        f"import app.lib.vnstock_data_alt, vnstock_data\nleft = {left_expr}\nright = {right_expr}",
        namespace,
    )
    assert inspect.signature(namespace["left"]) == inspect.signature(namespace["right"])


def test_vnstock_data_ui_helpers_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    module = _import("app.lib.vnstock_data_alt")

    module.show_doc(module.Market)
    module.show_api(module.Reference(), show_navigation=False)

    captured = capsys.readouterr()
    assert "Signature" in captured.out
    assert "API STRUCTURE TREE" in captured.out


def test_vnstock_alt_listing_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    listing_module = _import("app.lib.vnstock_alt.explorer.vci.listing")
    vnstock_alt = _import("app.lib.vnstock_alt")

    def fake_send_request(url: str, **kwargs):
        payload = kwargs.get("payload")
        payload_text = str(payload)

        if url.endswith("/price/symbols/getAll"):
            return [
                {
                    "symbol": "VCB",
                    "organName": "Vietcombank",
                    "type": "STOCK",
                    "board": "HSX",
                    "listingDate": "2009-06-30",
                    "id": 1,
                }
            ]
        if "getByGroup" in url:
            return [{"symbol": "VCB"}, {"symbol": "TCB"}]
        if "ListIcbCode" in payload_text:
            return {
                "data": {
                    "ListIcbCode": [
                        {
                            "icbCode": "8300",
                            "level": 2,
                            "icbName": "Ngan hang",
                            "enIcbName": "Banks",
                            "__typename": "IcbCode",
                        }
                    ],
                    "CompaniesListingInfo": [
                        {
                            "ticker": "VCB",
                            "icbCode1": "8000",
                            "icbCode2": "8300",
                            "icbCode3": "8350",
                            "icbCode4": "8351",
                            "__typename": "Company",
                        }
                    ],
                }
            }
        if "CompaniesListingInfo" in payload_text:
            return {
                "data": {
                    "CompaniesListingInfo": [
                        {
                            "ticker": "VCB",
                            "organName": "Vietcombank",
                            "enOrganName": "Vietcombank",
                            "icbName2": "Banks",
                            "enIcbName2": "Banks",
                            "icbName3": "Commercial Banks",
                            "enIcbName3": "Commercial Banks",
                            "icbName4": "Large Banks",
                            "enIcbName4": "Large Banks",
                            "comTypeCode": "STOCK",
                            "icbCode1": "8000",
                            "icbCode2": "8300",
                            "icbCode3": "8350",
                            "icbCode4": "8351",
                            "__typename": "Company",
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected request for {url}")

    monkeypatch.setattr(listing_module, "send_request", fake_send_request)

    listing = vnstock_alt.Listing(source="VCI")
    indices = listing.all_indices()
    assert {"symbol", "group", "full_name"}.issubset(indices.columns)

    all_symbols = listing.all_symbols()
    assert {"symbol", "organ_name"}.issubset(all_symbols.columns)

    industries = listing.industries_icb()
    assert {"icb_name", "en_icb_name", "icb_code", "level"}.issubset(industries.columns)

    by_industry = listing.symbols_by_industries()
    assert {"symbol", "organ_name", "icb_name2", "icb_name3", "icb_name4", "icb_code2"}.issubset(by_industry.columns)

    by_group = listing.symbols_by_group("ETF")
    assert len(by_group) > 0
    assert all(isinstance(symbol, str) and symbol for symbol in by_group.tolist())


def test_vnstock_alt_trading_price_board_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    trading_module = _import("app.lib.vnstock_alt.explorer.vci.trading")
    vnstock_alt = _import("app.lib.vnstock_alt")

    def fake_send_request(url: str, **kwargs):
        assert "getList" in url
        return [
            {
                "listingInfo": {
                    "symbol": "VCB",
                    "organName": "Vietcombank",
                    "board": "HSX",
                    "listedShare": 5589137101,
                    "charterCapital": 55891371010000,
                    "pe": 13.2,
                    "refPrice": 90000,
                },
                "bidAsk": {
                    "session": "A",
                    "time": "09:15",
                    "foreignBuyValue": 1000000,
                    "foreignSellValue": 500000,
                    "currentRoom": 10,
                    "totalRoom": 100,
                    "bidPrices": [],
                    "askPrices": [],
                },
                "matchPrice": {
                    "matchPrice": 91500,
                    "priceChangeRatio": 1.67,
                    "accumulatedValue": 250000000,
                    "time": "09:15",
                },
            }
        ]

    monkeypatch.setattr(trading_module.client, "send_request", fake_send_request)

    board = vnstock_alt.Trading(source="VCI").price_board(["VCB"])
    if isinstance(board.columns, pd.MultiIndex):
        assert ("listing", "symbol") in board.columns
        assert ("match", "match_price") in board.columns
    else:
        assert {"listing_symbol", "match_match_price"}.issubset(board.columns)


def test_vnstock_data_alt_vci_trading_history_and_flow_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    trading_module = _import("app.lib.vnstock_data_alt.explorer.vci.trading")

    def fake_fetch_data(self, endpoint: str, params: dict):
        assert params["timeFrame"] == "ONE_DAY"
        if endpoint == "price-history":
            return {
                "data": {
                    "content": [
                        {
                            "tradingDate": "2026-01-02",
                            "openPrice": 10.0,
                            "highestPrice": 11.0,
                            "lowestPrice": 9.5,
                            "closePrice": 10.5,
                            "totalMatchVolume": 100,
                            "totalMatchValue": 1000,
                            "totalDealVolume": 40,
                            "totalDealValue": 400,
                            "foreignBuyVolume": 140,
                            "foreignBuyValue": 1400,
                            "foreignSellVolume": 60,
                            "foreignSellValue": 600,
                            "foreignNetVolume": 80,
                            "foreignNetValue": 800,
                        }
                    ]
                }
            }
        if endpoint == "proprietary-history":
            return {
                "data": {
                    "content": [
                        {
                            "tradingDate": "2026-01-02",
                            "proprietaryBuyVolume": 70,
                            "proprietaryBuyValue": 700,
                            "proprietarySellVolume": 20,
                            "proprietarySellValue": 200,
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr(trading_module.Trading, "_fetch_data", fake_fetch_data)

    trading = trading_module.Trading(symbol="VCB", show_log=False)
    history = trading.price_history(start="2026-01-01", end="2026-01-31", get_all=True)
    foreign = trading.foreign_trade(start="2026-01-01", end="2026-01-31")
    prop = trading.prop_trade(start="2026-01-01", end="2026-01-31")

    assert {"matched_volume", "matched_value", "deal_volume", "deal_value"}.issubset(history.columns)
    assert {"fr_buy_volume", "fr_buy_value", "fr_sell_volume", "fr_sell_value", "fr_net_volume", "fr_net_value"}.issubset(history.columns)
    assert history.loc[0, "matched_value"] == 1000
    assert history.loc[0, "deal_value"] == 400
    assert history.loc[0, "fr_buy_value"] == 1400
    assert history.loc[0, "fr_sell_value"] == 600

    assert "matched_volume" not in foreign.columns
    assert "deal_value" not in foreign.columns
    assert {"fr_buy_volume", "fr_buy_value", "fr_sell_volume", "fr_sell_value", "fr_net_volume", "fr_net_value"}.issubset(foreign.columns)
    assert foreign.loc[0, "fr_buy_value"] == 1400
    assert foreign.loc[0, "fr_sell_value"] == 600
    assert foreign.loc[0, "fr_net_value"] == 800

    assert "matched_volume" not in prop.columns
    assert "deal_value" not in prop.columns
    assert {"prop_buy_volume", "prop_buy_value", "prop_sell_volume", "prop_sell_value"}.issubset(prop.columns)
    assert prop.loc[0, "prop_buy_value"] == 700
    assert prop.loc[0, "prop_sell_value"] == 200


def test_vnstock_alt_company_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    company_module = _import("app.lib.vnstock_alt.explorer.kbs.company")

    profile_payload = {
        "SB": "VCB",
        "SM": "<p>Profile</p>",
        "FD": "1963-04-01",
        "CC": 1000,
        "LD": "2009-06-30",
        "FV": 10000,
        "EX": "HSX",
        "LP": 100000,
        "VL": 123456789,
        "CTP": "CEO",
        "CTPP": "Chief Executive Officer",
        "IS": "Chief Inspector",
        "ISP": "Chief Inspector",
        "TC": "123456789",
        "TY": "Bank",
        "ADD": "1 Main St",
        "PHONE": "0123",
        "EMAIL": "info@example.com",
        "URL": "https://example.com",
        "HS": "<p>History</p>",
        "SFV": 1000,
        "KLCPLH": 2000,
        "AD": "2026-04-19",
        "Shareholders": [
            {"NM": "State Bank", "D": "2026-01-01", "V": 100, "OR": 74.8},
        ],
        "Leaders": [
            {"FD": "2026-01-01", "PN": "CEO", "NM": "Alice", "PO": "Chief Executive Officer", "PI": "VCB001"},
        ],
        "Subsidiaries": [
            {"D": "2026-01-01", "NM": "VCB Leasing", "CC": 500, "OR": 60.0, "CR": "VND"},
        ],
        "Ownership": [
            {"NM": "State", "OR": 74.8, "SH": 100, "D": "2026-01-01"},
        ],
        "CharterCapital": [
            {"D": "2025-01-01", "V": 1000, "C": "VND"},
        ],
        "LaborStructure": [
            {"Value": 10},
            {"Value": 20},
        ],
    }

    def fake_send_request(*args, **kwargs):
        url = kwargs.get("url") or (args[0] if args else "")
        if "/news/internal-trading/" in url:
            return [{"personName": "Alice", "action": "Buy"}]
        if "/event/" in url:
            return [{"eventName": "AGM", "eventDate": "2026-05-01"}]
        if "/news/" in url:
            return [{"head": "corp", "articleId": 1, "title": "Title", "publishTime": "2026-04-19", "url": "https://example.com/news"}]
        return profile_payload

    monkeypatch.setattr(company_module, "send_request", fake_send_request)

    company = company_module.Company("VCB")
    overview = company.overview()
    shareholders = company.shareholders()
    officers = company.officers()
    subsidiaries = company.subsidiaries()
    ownership = company.ownership()
    capital_history = company.capital_history()
    news = company.news()
    events = company.events()
    insider_trading = company.insider_trading()

    assert {"symbol", "business_model", "outstanding_shares", "company_type"}.issubset(overview.columns)
    assert {"name", "shares_owned", "ownership_percentage", "update_date"}.issubset(shareholders.columns)
    assert {"name", "position", "position_en", "owner_code", "from_date"}.issubset(officers.columns)
    assert {"name", "charter_capital", "ownership_percent", "currency", "type", "update_date"}.issubset(subsidiaries.columns)
    assert {"owner_type", "ownership_percentage", "shares_owned", "update_date"}.issubset(ownership.columns)
    assert {"date", "charter_capital", "currency"}.issubset(capital_history.columns)
    assert {"head", "article_id", "title", "publish_time", "url"}.issubset(news.columns)
    assert {"event_name", "event_date"}.issubset(events.columns)
    assert {"person_name", "action"}.issubset(insider_trading.columns)


def test_vnstock_alt_fund_details_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    fund_module = _import("app.lib.vnstock_alt.explorer.fmarket.fund")

    monkeypatch.setattr(
        fund_module.Fund,
        "listing",
        lambda self, fund_type="": pd.DataFrame({"short_name": ["SSISCA"]}),
    )

    fund = fund_module.Fund()
    monkeypatch.setattr(fund, "filter", lambda symbol="": pd.DataFrame({"id": [23], "shortName": [symbol]}))
    monkeypatch.setattr(fund, "nav_report", lambda fundId=23: pd.DataFrame({"date": ["2026-01-01"], "nav_per_unit": [12.3]}))
    monkeypatch.setattr(fund, "top_holding", lambda fundId=23: pd.DataFrame({"stock_code": ["FPT"], "net_asset_percent": [10.5]}))
    monkeypatch.setattr(fund, "industry_holding", lambda fundId=23: pd.DataFrame({"industry": ["Tech"], "net_asset_percent": [20.0]}))
    monkeypatch.setattr(fund, "asset_holding", lambda fundId=23: pd.DataFrame({"asset_type": ["Equity"], "asset_percent": [95.0]}))

    assert "short_name" in fund.details.nav_report("SSISCA").columns
    assert "short_name" in fund.details.top_holding("SSISCA").columns
    assert "short_name" in fund.details.industry_holding("SSISCA").columns
    assert "short_name" in fund.details.asset_holding("SSISCA").columns


def test_vnstock_alt_vnstock_chain_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    vnstock_alt = _import("app.lib.vnstock_alt")

    stock = vnstock_alt.Vnstock().stock(symbol="VCB", source="KBS")

    quote_df = pd.DataFrame(
        {
            "time": ["2026-01-02"],
            "open": [90000],
            "high": [92000],
            "low": [89500],
            "close": [91500],
            "volume": [3210000],
        }
    )
    finance_df = pd.DataFrame(
        {
            "ticker": ["VCB"],
            "yearReport": [2025],
            "lengthReport": [4],
            "Revenue": [15000000000000],
        }
    )

    monkeypatch.setattr(stock.quote.data_source, "history", lambda **kwargs: quote_df)
    for method in ["income_statement", "balance_sheet", "cash_flow", "ratio"]:
        monkeypatch.setattr(stock.finance.data_source, method, lambda **kwargs: finance_df)

    history = stock.quote.history(start="2026-01-01", end="2026-02-01", interval="1D")
    assert {"time", "open", "high", "low", "close", "volume"}.issubset(history.columns)

    assert {"ticker", "Revenue"}.issubset(stock.finance.income_statement(period="quarter", lang="en").columns)
    assert {"ticker", "Revenue"}.issubset(stock.finance.balance_sheet(period="quarter", lang="en").columns)
    assert {"ticker", "Revenue"}.issubset(stock.finance.cash_flow(period="quarter", lang="en").columns)
    assert {"ticker", "Revenue"}.issubset(stock.finance.ratio(period="quarter", lang="en").columns)
