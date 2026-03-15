'\nSchema mappings for UI module standardized output.\n'
_B=None
_A='ENUM_MAP'
from.core import standardize_columns,register_schemas
from.import company
from.import equity
from.import industry
from.import derivatives
from.import market
from.import insights
from.import fundamental
from.import macro
from.import fund
from.import events
from.import search
register_schemas(company.SCHEMA_MAP,company.STANDARD_COLUMNS,getattr(company,_A,_B))
register_schemas(equity.SCHEMA_MAP,equity.STANDARD_COLUMNS,getattr(equity,_A,_B))
register_schemas(industry.SCHEMA_MAP,industry.STANDARD_COLUMNS,getattr(industry,_A,_B))
register_schemas(derivatives.SCHEMA_MAP,derivatives.STANDARD_COLUMNS,getattr(derivatives,_A,_B))
register_schemas(market.SCHEMA_MAP,market.STANDARD_COLUMNS,getattr(market,_A,_B))
register_schemas(insights.SCHEMA_MAP,insights.STANDARD_COLUMNS,getattr(insights,_A,_B))
register_schemas(fundamental.SCHEMA_MAP,fundamental.STANDARD_COLUMNS,getattr(fundamental,_A,_B))
register_schemas(macro.SCHEMA_MAP,macro.STANDARD_COLUMNS,getattr(macro,_A,_B))
register_schemas(fund.SCHEMA_MAP,fund.STANDARD_COLUMNS,getattr(fund,_A,_B))
register_schemas(events.SCHEMA_MAP,events.STANDARD_COLUMNS,getattr(events,_A,_B))
register_schemas(search.SCHEMA_MAP,search.STANDARD_COLUMNS,getattr(search,_A,_B))
__all__=['standardize_columns']