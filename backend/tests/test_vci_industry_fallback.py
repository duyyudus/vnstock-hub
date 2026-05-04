from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.lib._vnstock_shared.common.vci_industry_fallback import (
    build_vci_industry_fallback,
    invalidate_vci_industry_fallback_cache,
)


SAMPLE_CRITERIA = [
    {
        "name": "sectorLv1",
        "conditionOptions": [
            {"value": "0001", "viName": "Dầu khí", "enName": "Oil & Gas"},
            {"value": "8301", "viName": "Ngân hàng", "enName": "Banks"},
            {"value": "1000", "viName": "Nguyên vật liệu", "enName": "Basic Materials"},
        ],
    },
    {
        "name": "sector",
        "conditionOptions": [
            {"value": "0500", "viName": "Dầu khí", "enName": "Oil & Gas"},
            {"value": "8300", "viName": "Ngân hàng", "enName": "Banks"},
            {"value": "1700", "viName": "Tài nguyên Cơ bản", "enName": "Basic Resources"},
        ],
    },
]

SAMPLE_SYMBOLS_BY_EXCHANGE = [
    {
        "symbol": "VCB",
        "board": "HSX",
        "type": "STOCK",
        "organName": "Vietcombank",
        "icbCode2": "8300",
        "id": 1,
    },
    {
        "symbol": "PVD",
        "board": "HSX",
        "type": "STOCK",
        "organName": "PV Drilling",
        "icbCode2": "0500",
        "id": 2,
    },
    {
        "symbol": "HPG",
        "board": "HSX",
        "type": "STOCK",
        "organName": "Hoa Phat",
        "icbCode2": "1700",
        "id": 3,
    },
    {
        "symbol": "CW1",
        "board": "HSX",
        "type": "CW",
        "organName": "Covered Warrant",
        "icbCode2": "8300",
        "id": 4,
    },
]


@pytest.fixture(autouse=True)
def _reset_vci_fallback_cache() -> None:
    invalidate_vci_industry_fallback_cache()


def _sample_symbols_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "VCB", "organ_name": "Vietcombank", "type": "STOCK", "icb_code2": "8300"},
            {"symbol": "PVD", "organ_name": "PV Drilling", "type": "STOCK", "icb_code2": "0500"},
            {"symbol": "HPG", "organ_name": "Hoa Phat", "type": "STOCK", "icb_code2": "1700"},
            {"symbol": "CW1", "organ_name": "Covered Warrant", "type": "CW", "icb_code2": "8300"},
        ]
    )


def _fake_vci_send_request(url: str, **_: object):
    if "price/symbols/getAll" in url:
        return SAMPLE_SYMBOLS_BY_EXCHANGE
    if "graphql" in url:
        return {}
    raise AssertionError(f"Unexpected request for {url}")


def test_build_vci_industry_fallback_reconstructs_special_family_codes() -> None:
    fallback = build_vci_industry_fallback(
        _sample_symbols_frame(),
        criteria_raw=SAMPLE_CRITERIA,
    )

    industries = fallback.industries_icb
    assert {"icb_name", "en_icb_name", "icb_code", "level"} == set(industries.columns)

    oil_and_gas_level1 = industries[(industries["icb_code"] == "0001") & (industries["level"] == 1)]
    banks_level1 = industries[(industries["icb_code"] == "8301") & (industries["level"] == 1)]
    banks_level2 = industries[(industries["icb_code"] == "8300") & (industries["level"] == 2)]

    assert oil_and_gas_level1.iloc[0]["icb_name"] == "Dầu khí"
    assert banks_level1.iloc[0]["icb_name"] == "Ngân hàng"
    assert banks_level2.iloc[0]["en_icb_name"] == "Banks"

    symbols = fallback.symbols_by_level2
    assert list(symbols["symbol"]) == ["HPG", "PVD", "VCB"]
    assert symbols.loc[symbols["symbol"] == "PVD", "icb_code1"].iloc[0] == "0001"
    assert symbols.loc[symbols["symbol"] == "VCB", "icb_code1"].iloc[0] == "8301"
    assert symbols.loc[symbols["symbol"] == "HPG", "icb_name2"].iloc[0] == "Tài nguyên Cơ bản"


def test_vnstock_alt_listing_falls_back_when_vci_graphql_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    listing_module = importlib.import_module("app.lib.vnstock_alt.explorer.vci.listing")
    monkeypatch.setattr(listing_module, "send_request", _fake_vci_send_request)
    monkeypatch.setattr(
        "app.lib._vnstock_shared.common.vci_industry_fallback.get_vci_screener_criteria",
        lambda **_: SAMPLE_CRITERIA,
    )

    listing = importlib.import_module("app.lib.vnstock_alt").Listing(source="VCI")

    industries = listing.industries_icb()
    assert {"icb_name", "en_icb_name", "icb_code", "level"} == set(industries.columns)
    assert {"0001", "0500", "8301", "8300"}.issubset(set(industries["icb_code"]))

    by_industry = listing.symbols_by_industries()
    assert {"symbol", "organ_name", "icb_name3", "icb_name2", "icb_name4", "com_type_code", "icb_code1", "icb_code2", "icb_code3", "icb_code4"} == set(by_industry.columns)
    assert by_industry.loc[by_industry["symbol"] == "VCB", "icb_name2"].iloc[0] == "Ngân hàng"
    assert by_industry.loc[by_industry["symbol"] == "PVD", "icb_code1"].iloc[0] == "0001"
    assert by_industry["icb_name3"].isna().all()
    assert by_industry["icb_code3"].isna().all()

    by_industry_en = listing.symbols_by_industries(lang="en")
    assert by_industry_en.loc[by_industry_en["symbol"] == "VCB", "icb_name2"].iloc[0] == "Banks"


def test_vnstock_data_alt_listing_falls_back_when_vci_graphql_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    listing_module = importlib.import_module("app.lib.vnstock_data_alt.explorer.vci.listing")
    monkeypatch.setattr(listing_module, "send_request", _fake_vci_send_request)
    monkeypatch.setattr(
        "app.lib._vnstock_shared.common.vci_industry_fallback.get_vci_screener_criteria",
        lambda **_: SAMPLE_CRITERIA,
    )

    listing = importlib.import_module("app.lib.vnstock_data_alt").Listing(source="VCI")

    industries = listing.industries_icb()
    assert {"icb_name", "en_icb_name", "icb_code", "level"} == set(industries.columns)
    assert {"0001", "0500", "8301", "8300"}.issubset(set(industries["icb_code"]))

    by_industry = listing.symbols_by_industries()
    assert {"symbol", "organ_name", "com_type_code", "icb_level", "icb_code", "icb_name"} == set(by_industry.columns)
    vcb_rows = by_industry[by_industry["symbol"] == "VCB"]
    assert list(vcb_rows["icb_level"]) == [1, 2]
    assert list(vcb_rows["icb_code"]) == ["8301", "8300"]
    assert list(vcb_rows["icb_name"]) == ["Ngân hàng", "Ngân hàng"]

    by_industry_en = listing.symbols_by_industries(lang="en")
    assert by_industry_en[by_industry_en["symbol"] == "PVD"]["icb_name"].tolist() == ["Oil & Gas", "Oil & Gas"]


def test_stocks_service_uses_kbs_industry_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    service = importlib.import_module("app.services.vnstock_service.stocks").StocksService(
        metadata=MagicMock(),
        history=MagicMock(),
    )

    class FakeListing:
        def __init__(self, source: str):
            assert source == "KBS"

        def symbols_by_industries(self):
            return pd.DataFrame(
                [
                    {"symbol": "VCB", "industry_code": 11, "industry_name": "Ngân hàng"},
                    {"symbol": "PVD", "industry_code": 10, "industry_name": "Khai khoáng"},
                    {"symbol": "HPG", "industry_code": 21, "industry_name": "Vật liệu xây dựng"},
                ]
            )

    monkeypatch.setitem(sys.modules, "vnstock", SimpleNamespace(Listing=FakeListing))

    with patch("app.services.vnstock_service.stocks.api_circuit_breaker") as breaker:
        breaker.can_proceed.return_value = True

        industries = service._fetch_industries_sync()
        assert {"KBS-10", "KBS-11", "KBS-21"}.issubset(set(industries["icb_code"]))

        mapping = service._get_or_fetch_industry_mapping()
        assert mapping == {
            "HPG": "Vật liệu xây dựng",
            "PVD": "Khai khoáng",
            "VCB": "Ngân hàng",
        }

        captured: dict[str, object] = {}

        def _capture_symbols(symbols: list[str], limit: int):
            captured["symbols"] = symbols
            captured["limit"] = limit
            return []

        monkeypatch.setattr(service, "_fetch_symbols_data", _capture_symbols)
        result = service._fetch_industry_data("Ngân hàng", 10)

    assert result == []
    assert captured == {"symbols": ["VCB"], "limit": 10}
