"""
Company schemas mapping definitions.
"""

SCHEMA_MAP = {
    "company.info": {
        "kbs": {},
    },
    "company.shareholders": {
        "kbs": {},
    },
    "company.officers": {
        "kbs": {},
    },
    "company.subsidiaries": {
        "kbs": {},
    },
    "company.ownership": {
        "kbs": {},
    },
    "company.capital_history": {
        "kbs": {},
    },
    "company.news": {
        "kbs": {},
    },
    "company.events": {
        "kbs": {},
    },
    "company.insider_trading": {
        "kbs": {},
    },
    "company.margin_ratio": {
        "kbs": {
            "CompanyCode": "broker_code",
            "Name": "broker_name",
            "MarginRate": "margin_rate",
            "PrevMarginRate": "prev_margin_rate",
            "ClosedDate": "updated_at",
            "MarginPer": "margin_per",
        },
    },
}

STANDARD_COLUMNS = {
    "company.info": [
        "business_model",
        "symbol",
        "founded_date",
        "charter_capital",
        "charter_capital_vnd",
        "number_of_employees",
        "listing_date",
        "par_value",
        "exchange",
        "listing_price",
        "listed_volume",
        "ceo_name",
        "ceo_position",
        "inspector_name",
        "inspector_position",
        "establishment_license",
        "business_code",
        "tax_id",
        "auditor",
        "company_type",
        "address",
        "phone",
        "fax",
        "email",
        "website",
        "branches",
        "history",
        "outstanding_shares",
        "as_of_date",
    ],
    "company.shareholders": [
        "name",
        "update_date",
        "shares_owned",
        "ownership_percentage",
    ],
    "company.officers": [
        "from_date",
        "position",
        "name",
        "position_en",
        "owner_code",
    ],
    "company.subsidiaries": [
        "update_date",
        "name",
        "charter_capital",
        "ownership_percent",
        "currency",
        "type",
    ],
    "company.ownership": [
        "owner_type",
        "ownership_percentage",
        "shares_owned",
        "update_date",
    ],
    "company.capital_history": [
        "date",
        "charter_capital",
        "currency",
    ],
    "company.news": [
        "head",
        "article_id",
        "title",
        "publish_time",
        "url",
    ],
    "company.margin_ratio": [
        "broker_code",
        "broker_name",
        "margin_rate",
        "prev_margin_rate",
        "margin_per",
        "updated_at",
    ],
}

ENUM_MAP = {
    "type": {
        "công ty con": "Subsidiary",
        "công ty liên kết": "Affiliate",
    }
}
