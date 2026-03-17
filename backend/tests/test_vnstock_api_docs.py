from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.capture_vnstock_api_samples import collect_probe_definitions, validate_snapshot
from scripts.generate_vnstock_api_docs import build_docs_metadata, render_docs
from scripts.vnstock_api_docs_common import BACKEND_ROOT, LIVE_ROOT


@pytest.fixture(scope="module")
def docs_metadata() -> dict:
    return build_docs_metadata()


def test_build_docs_metadata_contains_representative_surfaces(docs_metadata: dict) -> None:
    packages = {package["package"]: package for package in docs_metadata["packages"]}

    vnstock_alt_exports = {export["name"]: export for export in packages["vnstock_alt"]["exports"]}
    vnstock_data_exports = {export["name"]: export for export in packages["vnstock_data_alt"]["exports"]}

    assert "Quote" in vnstock_alt_exports
    assert "Market" in vnstock_data_exports

    alt_quote_methods = {method["name"]: method for method in vnstock_alt_exports["Quote"]["methods"]}
    alt_trading_methods = {method["name"]: method for method in vnstock_alt_exports["Trading"]["methods"]}
    data_market_methods = {method["name"]: method for method in vnstock_data_exports["Market"]["methods"]}
    data_finance_methods = {method["name"]: method for method in vnstock_data_exports["Finance"]["methods"]}
    alt_listing_methods = {method["name"]: method for method in vnstock_alt_exports["Listing"]["methods"]}
    data_trading_methods = {method["name"]: method for method in vnstock_data_exports["Trading"]["methods"]}

    assert "history" in alt_quote_methods
    assert alt_quote_methods["history"]["raw_outputs"]
    history_params = {param["name"]: param for param in alt_quote_methods["history"]["parameters"]}
    assert history_params["start"]["example"] == '"2024-01-01"'
    assert history_params["start"]["observed_example"] == "2025-03-01"
    assert history_params["end"]["example"] == '"2024-04-18"'
    assert "1m" in history_params["interval"]["accepted_values"]
    assert "equity" in data_market_methods
    assert data_finance_methods["balance_sheet"]["signature"] != data_finance_methods["balance_sheet"]["declared_signature"]
    assert data_finance_methods["balance_sheet"]["signature_hint_source"] == "kbs"
    assert alt_listing_methods["symbols_by_exchange"]["raw_outputs"][0]["raw_columns"] == [
        "symbol",
        "organ_name",
        "en_organ_name",
        "exchange",
        "type",
        "id",
    ]
    assert "price_board" in alt_trading_methods
    assert "foreign_trade" not in alt_trading_methods
    assert "history" not in alt_trading_methods
    assert "price_depth" not in alt_quote_methods
    assert "history" not in {method["name"] for method in vnstock_alt_exports["Listing"]["methods"]}
    assert data_trading_methods["price_board"]["raw_outputs"][1]["coverage"] == "declared"


def test_collect_probe_definitions_auto_expands_manifest() -> None:
    manifest_only = collect_probe_definitions(include_auto=False)
    auto_expanded = collect_probe_definitions(include_auto=True)

    manifest_keys = {
        (
            probe["package"],
            probe["class_path"],
            probe["method"],
            probe.get("source"),
        )
        for probe in manifest_only
    }
    expected_backend_manifest_keys = {
        ("vnstock_alt", "app.lib.vnstock_alt.api.listing.Listing", "all_indices", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.listing.Listing", "industries_icb", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.listing.Listing", "symbols_by_industries", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.listing.Listing", "symbols_by_group", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.listing.Listing", "all_symbols", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.trading.Trading", "price_board", "kbs"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.trading.Trading", "price_board", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.quote.Quote", "history", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.financial.Finance", "income_statement", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.financial.Finance", "balance_sheet", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.financial.Finance", "cash_flow", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.financial.Finance", "ratio", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.company.Company", "overview", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.company.Company", "shareholders", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.company.Company", "officers", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.api.company.Company", "subsidiaries", "vci"),
        ("vnstock_alt", "app.lib.vnstock_alt.explorer.fmarket.fund.Fund", "listing", "fmarket"),
        ("vnstock_alt", "app.lib.vnstock_alt.explorer.fmarket.fund.Fund", "nav_report", "fmarket"),
        ("vnstock_alt", "app.lib.vnstock_alt.explorer.fmarket.fund.Fund", "top_holding", "fmarket"),
        ("vnstock_alt", "app.lib.vnstock_alt.explorer.fmarket.fund.Fund", "industry_holding", "fmarket"),
        ("vnstock_alt", "app.lib.vnstock_alt.explorer.fmarket.fund.Fund", "asset_holding", "fmarket"),
        ("vnstock_data_alt", "app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund", "listing", "fmarket"),
        ("vnstock_data_alt", "app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund", "nav_report", "fmarket"),
        ("vnstock_data_alt", "app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund", "top_holding", "fmarket"),
        ("vnstock_data_alt", "app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund", "industry_holding", "fmarket"),
        ("vnstock_data_alt", "app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund", "asset_holding", "fmarket"),
    }

    assert len(manifest_only) == 29
    assert expected_backend_manifest_keys <= manifest_keys
    assert len(auto_expanded) > len(manifest_only)

    auto_by_schema = {probe.get("schema_key") for probe in auto_expanded if probe.get("schema_key")}
    assert "company.info" in auto_by_schema
    assert "market.index.ohlcv" in auto_by_schema
    assert "market.equity.quote" in auto_by_schema

    quote_history_sources = {
        probe["source"]
        for probe in auto_expanded
        if probe["package"] == "vnstock_alt"
        and probe["class_path"] == "app.lib.vnstock_alt.api.quote.Quote"
        and probe["method"] == "history"
    }
    assert {"kbs", "msn", "vci"} <= quote_history_sources

    price_board_probes = [
        probe
        for probe in auto_expanded
        if probe["package"] == "vnstock_alt"
        and probe["class_path"] == "app.lib.vnstock_alt.api.trading.Trading"
        and probe["method"] == "price_board"
    ]
    assert {probe["source"] for probe in price_board_probes} >= {"kbs", "vci"}
    for probe in price_board_probes:
        assert probe["method_kwargs"]["symbols_list"] == ["VCB", "TCB"]

    fund_probes = [
        probe
        for probe in manifest_only
        if probe["class_path"] in {
            "app.lib.vnstock_alt.explorer.fmarket.fund.Fund",
            "app.lib.vnstock_data_alt.explorer.fmarket.fund.Fund",
        }
        and probe["method"] in {
            "listing",
            "nav_report",
            "top_holding",
            "industry_holding",
            "asset_holding",
        }
    ]
    assert {
        (probe["package"], probe["method"])
        for probe in fund_probes
    } >= {
        ("vnstock_alt", "listing"),
        ("vnstock_alt", "nav_report"),
        ("vnstock_alt", "top_holding"),
        ("vnstock_alt", "industry_holding"),
        ("vnstock_alt", "asset_holding"),
        ("vnstock_data_alt", "listing"),
        ("vnstock_data_alt", "nav_report"),
        ("vnstock_data_alt", "top_holding"),
        ("vnstock_data_alt", "industry_holding"),
        ("vnstock_data_alt", "asset_holding"),
    }


def test_schema_metadata_matches_registered_schema_contract(docs_metadata: dict) -> None:
    schemas = {schema["schema_key"]: schema for schema in docs_metadata["schemas"]}

    company_info = schemas["company.info"]
    assert company_info["class_name"] == "CompanyReference"
    assert company_info["default_route"]["source"] == "vci"
    assert company_info["normalized_output"]["columns"] == [
        "symbol",
        "name",
        "short_name",
        "exchange",
        "sector",
        "industry",
        "profile",
        "history",
        "num_employees",
        "founded_date",
        "listing_date",
        "charter_capital",
        "issued_share",
        "website",
        "address",
        "phone",
        "email",
        "tax_id",
    ]

    ohlcv = schemas["market.equity.ohlcv"]
    assert ohlcv["class_name"] == "EquityMarket"
    assert ohlcv["return_type"] == "pd.DataFrame"
    assert ohlcv["signature"] == "(start: str = None, end: str = None, interval: str = None, count_back: int = None) -> pd.DataFrame"
    assert {item["source"] for item in ohlcv["raw_outputs"]} >= {"kbs", "msn"}
    assert ohlcv["live_samples"]

    futures_summary = schemas["market.futures.summary"]
    futures_params = {param["name"]: param for param in futures_summary["parameters"]}
    assert futures_params["symbol"]["observed_example"] == "VN30F1M"

    odd_lot = schemas["market.equity.odd_lot"]
    odd_lot_params = {param["name"]: param for param in odd_lot["parameters"]}
    assert odd_lot_params["symbols_list"]["observed_example"] == "['VCB']"

    search_symbol = schemas["search.symbol"]
    search_symbol_params = {param["name"]: param for param in search_symbol["parameters"]}
    assert search_symbol_params["locale"]["observed_example"] == "omitted in live probe"

    events_calendar = schemas["events.calendar"]
    event_params = {param["name"]: param for param in events_calendar["parameters"]}
    assert "dividend" in event_params["event_type"]["accepted_values"]
    assert event_params["event_type"]["observed_example"] == "omitted in live probe"

    screener_filter = schemas["insights.screener.filter"]
    assert {param["name"] for param in screener_filter["parameters"]} == {"limit"}

    observed_schema_samples = sum(1 for schema in schemas.values() if schema["live_samples"])
    assert observed_schema_samples >= 10


def test_render_docs_creates_expected_pages(tmp_path: Path, docs_metadata: dict) -> None:
    render_docs(docs_metadata, output_root=tmp_path)

    assert (tmp_path / "index.md").exists()
    assert (tmp_path / "coverage.md").exists()
    assert (tmp_path / "packages" / "vnstock_alt" / "quote.md").exists()
    assert (tmp_path / "packages" / "vnstock_data_alt" / "market.md").exists()
    assert (tmp_path / "schemas" / "company-info.md").exists()
    assert (tmp_path / "live-samples" / "index.md").exists()
    assert (tmp_path / "metadata.json").exists()
    schema_page = (tmp_path / "schemas" / "market-equity-ohlcv.md").read_text()
    schema_index = (tmp_path / "schemas" / "index.md").read_text()
    package_index = (tmp_path / "packages" / "vnstock_data_alt" / "index.md").read_text()
    live_index = (tmp_path / "live-samples" / "index.md").read_text()
    fund_page = (tmp_path / "packages" / "vnstock_alt" / "fund.md").read_text()
    quote_page = (tmp_path / "packages" / "vnstock_alt" / "quote.md").read_text()
    alt_trading_page = (tmp_path / "packages" / "vnstock_alt" / "trading.md").read_text()
    trading_page = (tmp_path / "packages" / "vnstock_data_alt" / "trading.md").read_text()
    assert "Declared signature" in schema_page
    assert "## Source details" in schema_page
    assert "### Source `kbs`" in schema_page
    assert "#### Raw output contract" in schema_page
    assert "#### Normalized output schema" in schema_page
    assert "#### Live-observed sample" in schema_page
    assert "- Captured at:" in schema_page
    assert "## market" in schema_index
    assert "### equity" in schema_index
    assert "## Provider Adapters" in package_index
    assert "## Unified UI Namespaces" in package_index
    assert "## vnstock_alt" in live_index or "## vnstock_data_alt" in live_index
    assert "\nParameters\n----------\n" not in fund_page
    assert "\nReturns\n-------\n" not in fund_page
    assert "\nTham số:\n----------\n" not in fund_page
    assert "\nTrả về:\n-------\n" not in fund_page
    assert "| Name | Kind | Required | Default | Annotation | Example | Observed example | Accepted values | Description |" in quote_page
    assert '`"2024-01-01"`' in quote_page
    assert "`2025-03-01`" in quote_page
    assert "`1m`, `5m`, `15m`, `30m`, `1H`, `D`, `1W`, `1M`" in quote_page
    assert "| `start` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` | `2025-03-01` |" in trading_page
    assert "### price_board" in alt_trading_page
    assert "### foreign_trade" not in alt_trading_page
    assert '"symbol": "V"' not in alt_trading_page


def test_live_sampled_package_methods_always_render_observed_example_column(docs_metadata: dict, tmp_path: Path) -> None:
    render_docs(docs_metadata, output_root=tmp_path)

    violations: list[tuple[str, str, str, str]] = []
    packages_root = tmp_path / "packages"
    for package_dir in sorted(packages_root.iterdir()):
        if not package_dir.is_dir():
            continue
        for md_path in sorted(package_dir.glob("*.md")):
            if md_path.name == "index.md":
                continue
            text = md_path.read_text()
            parts = re.split(r"^### ([^\n]+)\n", text, flags=re.M)
            if len(parts) < 3:
                continue
            for index in range(1, len(parts), 2):
                method_name = parts[index].strip()
                body = parts[index + 1]
                if "Live-observed sample" not in body or "- Captured at:" not in body:
                    continue
                header_match = re.search(r"^\| Name \|.*$", body, flags=re.M)
                if not header_match:
                    violations.append((package_dir.name, md_path.name, method_name, "missing parameter table"))
                    continue
                header = header_match.group(0)
                if "Observed example" not in header:
                    violations.append((package_dir.name, md_path.name, method_name, header))

    assert not violations, f"Missing Observed example column for live-sampled methods: {violations}"


def test_architecture_page_is_scoped_to_vendored_packages_only() -> None:
    architecture_page = (BACKEND_ROOT / "docs" / "architecture.md").read_text()

    assert "# Vendored Vnstock Package Architecture" in architecture_page
    assert "backend/app/lib" in architecture_page
    assert "vendored packages under" in architecture_page
    assert "It is not a map of the FastAPI application" in architecture_page
    assert "dependency map among the vendored packages only" in architecture_page
    assert "## Out Of Scope" in architecture_page


def test_live_snapshots_validate() -> None:
    snapshot_paths = sorted(LIVE_ROOT.glob("*.json"))
    assert snapshot_paths

    for path in snapshot_paths:
        if path.name == "index.json":
            continue
        payload = json.loads(path.read_text())
        validate_snapshot(payload)


def test_mkdocs_build_smoke(docs_metadata: dict, tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")

    render_docs(docs_metadata, output_root=BACKEND_ROOT / "docs" / "generated")
    site_dir = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "-f",
            str(BACKEND_ROOT / "mkdocs.yml"),
            "-d",
            str(site_dir),
        ],
        cwd=BACKEND_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (site_dir / "index.html").exists()
    assert (site_dir / "generated" / "index.html").exists()
    assert (site_dir / "architecture" / "index.html").exists()
