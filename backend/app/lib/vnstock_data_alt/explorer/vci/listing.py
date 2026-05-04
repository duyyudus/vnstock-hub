"""Listing module."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd

from app.lib._vnstock_shared.common.vci_industry_fallback import build_vci_industry_fallback
from app.lib._vnstock_shared.compat import agg_execution
from app.lib._vnstock_shared.core.utils.logger import get_logger
from app.lib._vnstock_shared.core.utils.parser import camel_to_snake
from app.lib._vnstock_shared.core.utils.transform import drop_cols_by_pattern, reorder_cols
from app.lib.vnstock_alt.explorer.vci.const import _GRAPHQL_URL, _GROUP_CODE, _TRADING_URL
from app.lib.vnstock_data_alt.core.registry import ProviderRegistry
from app.lib.vnstock_data_alt.core.utils.client import send_request
from app.lib.vnstock_data_alt.core.utils.user_agent import get_headers


logger = get_logger(__name__)


class Listing:
    """Cấu hình truy cập dữ liệu lịch sử giá chứng khoán từ VCI."""

    def __init__(self, random_agent: bool = False, show_log: bool = False):
        self.data_source = "VCI"
        self.base_url = _TRADING_URL
        self.headers = get_headers(data_source=self.data_source, random_agent=random_agent)
        self.show_log = show_log
        if not show_log:
            logger.setLevel("CRITICAL")

    @agg_execution("VCI.ext")
    def all_symbols(self, show_log: bool = False, to_df: bool = True):
        df = self.symbols_by_exchange(show_log=show_log, to_df=True)
        df = df.query('type == "STOCK"').reset_index(drop=True)
        df = df[["symbol", "organ_name"]]
        if to_df:
            return df
        return df.to_json(orient="records")

    @agg_execution("VCI.ext")
    def symbols_by_industries(self, lang: str = "vi", show_log: bool = False, to_df: bool = True):
        if lang not in ["vi", "en"]:
            raise ValueError("Tham số lang phải là 'vi' hoặc 'en'.")

        # Deprecated/blocked as of 2026-05-04 for CompaniesListingInfo.
        payload = json.loads(
            '{"query":"{\\n  CompaniesListingInfo {\\n    ticker\\n    organName\\n    enOrganName\\n    '
            'icbName3\\n    enIcbName3\\n    icbName2\\n    enIcbName2\\n    icbName4\\n    enIcbName4\\n    '
            'comTypeCode\\n    icbCode1\\n    icbCode2\\n    icbCode3\\n    icbCode4\\n    __typename\\n  }\\n}'
            '\\n","variables":{}}'
        )
        json_data = send_request(
            url=_GRAPHQL_URL,
            headers=self.headers,
            method="POST",
            payload=payload,
            show_log=show_log,
        )

        companies_listing = _extract_graphql_rows(json_data, "CompaniesListingInfo")
        if not companies_listing:
            df = self._fallback_symbols_by_industries(lang=lang, show_log=show_log)
        else:
            if show_log:
                logger.info("Truy xuất thành công dữ liệu danh sách phân ngành icb.")
            df = pd.DataFrame(companies_listing)
            df.columns = [camel_to_snake(col) for col in df.columns]
            df = df.drop(columns=["__typename"])
            df = df.rename(columns={"ticker": "symbol"})
            df = self._to_long_form_symbols_by_industries(df, lang=lang)

        df.source = "VCI"
        if to_df:
            return df
        return df.to_json(orient="records")

    @agg_execution("VCI.ext")
    def symbols_by_exchange(self, lang: str = "vi", show_log: bool = False, to_df: bool = True):
        if lang not in ["vi", "en"]:
            raise ValueError("Tham số lang phải là 'vi' hoặc 'en'.")

        # Deprecated/blocked as of 2026-05-04: VCI can return 403 HTML here.
        json_data = send_request(
            url=self.base_url + "/price/symbols/getAll",
            headers=self.headers,
            method="GET",
            payload=None,
            show_log=show_log,
        )
        if not json_data:
            raise ValueError("Không tìm thấy dữ liệu. Vui lòng kiểm tra lại.")

        if show_log:
            logger.info("Truy xuất dữ liệu thành công cho %s mã.", len(json_data))

        df = pd.DataFrame(json_data)
        df.columns = [camel_to_snake(col) for col in df.columns]
        df = df.rename(columns={"board": "exchange"})
        df = reorder_cols(df, ["symbol", "exchange", "type"], position="first")
        df = df.drop(columns=["id"])

        if lang == "vi":
            df = drop_cols_by_pattern(df, ["en_"])
        else:
            df = df.drop(columns=["organ_name", "organ_short_name"])
            df.columns = [col.replace("en_", "") for col in df.columns]

        df.source = "VCI"
        if to_df:
            return df
        return df.to_json(orient="records")

    @agg_execution("VCI.ext")
    def industries_icb(self, show_log: bool = False, to_df: bool = True):
        # Deprecated/blocked as of 2026-05-04 for ListIcbCode/CompaniesListingInfo.
        payload = json.loads(
            '{"query":"query Query {\\n  ListIcbCode {\\n    icbCode\\n    level\\n    icbName\\n    enIcbName'
            '\\n    __typename\\n  }\\n  CompaniesListingInfo {\\n    ticker\\n    icbCode1\\n    icbCode2\\n    '
            'icbCode3\\n    icbCode4\\n    __typename\\n  }\\n}","variables":{}}'
        )
        json_data = send_request(
            url=_GRAPHQL_URL,
            headers=self.headers,
            method="POST",
            payload=payload,
            show_log=show_log,
        )

        icb_rows = _extract_graphql_rows(json_data, "ListIcbCode")
        if not icb_rows:
            df = self._fallback_industries_icb(show_log=show_log)
        else:
            if show_log:
                logger.info("Truy xuất thành công dữ liệu danh sách phân ngành icb.")
            df = pd.DataFrame(icb_rows)
            df.columns = [camel_to_snake(col) for col in df.columns]
            df = df.drop(columns=["__typename"])
            df = df[["icb_name", "en_icb_name", "icb_code", "level"]]

        df.source = "VCI"
        if to_df:
            return df
        return df.to_json(orient="records")

    @agg_execution("VCI.ext")
    def symbols_by_group(self, group: str = "VN30", show_log: bool = False, to_df: bool = True):
        if group not in _GROUP_CODE:
            raise ValueError(f"Invalid group. Group must be in {_GROUP_CODE}")

        # Deprecated/blocked as of 2026-05-04: VCI can return 403 HTML here.
        json_data = send_request(
            url=self.base_url + f"/price/symbols/getByGroup?group={group}",
            headers=self.headers,
            method="GET",
            payload=None,
            show_log=show_log,
        )
        if show_log:
            logger.info("Truy xuất thành công dữ liệu danh sách mã CP theo nhóm.")

        df = pd.DataFrame(json_data)
        if to_df:
            if not json_data:
                raise ValueError("JSON data is empty or not provided.")
            df.source = "VCI"
            return df["symbol"]
        return df.to_json(orient="records")

    @agg_execution("VCI.ext")
    def all_future_indices(self, show_log: bool = False, to_df: bool = True):
        return self.symbols_by_group(group="FU_INDEX", show_log=show_log, to_df=to_df)

    @agg_execution("VCI.ext")
    def all_government_bonds(self, show_log: bool = False, to_df: bool = True):
        return self.symbols_by_group(group="FU_BOND", show_log=show_log, to_df=to_df)

    @agg_execution("VCI.ext")
    def all_covered_warrant(self, show_log: bool = False, to_df: bool = True):
        return self.symbols_by_group(group="CW", show_log=show_log, to_df=to_df)

    @agg_execution("VCI.ext")
    def all_bonds(self, show_log: bool = False, to_df: bool = True):
        return self.symbols_by_group(group="BOND", show_log=show_log, to_df=to_df)

    def _fallback_industries_icb(self, show_log: bool = False) -> pd.DataFrame:
        fallback = self._build_industry_fallback(show_log=show_log)
        return fallback.industries_icb.copy()

    def _fallback_symbols_by_industries(self, lang: str, show_log: bool = False) -> pd.DataFrame:
        fallback = self._build_industry_fallback(show_log=show_log)
        base_df = fallback.symbols_by_level2.copy()
        name_column = "icb_name2" if lang == "vi" else "en_icb_name2"
        family_name_column = "icb_name1" if lang == "vi" else "en_icb_name1"
        organ_name_column = "organ_name" if lang == "vi" else "en_organ_name"

        records: list[dict[str, object]] = []
        for row in base_df.itertuples(index=False):
            if row.icb_code1:
                records.append(
                    {
                        "symbol": row.symbol,
                        "organ_name": getattr(row, organ_name_column),
                        "com_type_code": row.com_type_code,
                        "icb_level": 1,
                        "icb_code": row.icb_code1,
                        "icb_name": getattr(row, family_name_column),
                    }
                )
            records.append(
                {
                    "symbol": row.symbol,
                    "organ_name": getattr(row, organ_name_column),
                    "com_type_code": row.com_type_code,
                    "icb_level": 2,
                    "icb_code": row.icb_code2,
                    "icb_name": getattr(row, name_column),
                }
            )

        df = pd.DataFrame(records, columns=["symbol", "organ_name", "com_type_code", "icb_level", "icb_code", "icb_name"])
        if not df.empty:
            df = df.sort_values(by=["symbol", "icb_level"]).reset_index(drop=True)
        if show_log:
            logger.warning("VCI industry GraphQL returned empty payload; using reconstructed listing fallback.")
        return df

    def _build_industry_fallback(self, show_log: bool = False):
        symbols_df = self.symbols_by_exchange(show_log=show_log, to_df=True)
        return build_vci_industry_fallback(symbols_df, random_agent=False, show_log=show_log)

    def _to_long_form_symbols_by_industries(self, df: pd.DataFrame, *, lang: str) -> pd.DataFrame:
        organ_name_column = "organ_name" if lang == "vi" else "en_organ_name"
        rows: list[pd.DataFrame] = []
        level_mappings = [
            (1, "icb_code1", None),
            (2, "icb_code2", "icb_name2" if lang == "vi" else "en_icb_name2"),
            (3, "icb_code3", "icb_name3" if lang == "vi" else "en_icb_name3"),
            (4, "icb_code4", "icb_name4" if lang == "vi" else "en_icb_name4"),
        ]

        for level, code_column, name_column in level_mappings:
            if code_column not in df.columns:
                continue
            level_df = df[["symbol", organ_name_column, "com_type_code", code_column]].copy()
            level_df["icb_level"] = level
            level_df = level_df.rename(columns={organ_name_column: "organ_name", code_column: "icb_code"})
            if name_column and name_column in df.columns:
                level_df["icb_name"] = df[name_column]
            else:
                level_df["icb_name"] = None
            rows.append(level_df)

        if not rows:
            return pd.DataFrame(columns=["symbol", "organ_name", "com_type_code", "icb_level", "icb_code", "icb_name"])

        long_df = pd.concat(rows, ignore_index=True)
        long_df = long_df[long_df["icb_code"].notna() & (long_df["icb_code"] != "")]
        return long_df[["symbol", "organ_name", "com_type_code", "icb_level", "icb_code", "icb_name"]].sort_values(
            by=["symbol", "icb_level"]
        ).reset_index(drop=True)


def _extract_graphql_rows(json_data: object, field_name: str) -> list[dict]:
    if not isinstance(json_data, dict):
        return []
    data = json_data.get("data")
    if not isinstance(data, dict):
        return []
    rows = data.get(field_name)
    if not isinstance(rows, list):
        return []
    return rows


ProviderRegistry.register("listing", "vci", Listing)
