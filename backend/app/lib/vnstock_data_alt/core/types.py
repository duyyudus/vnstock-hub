"\nType definitions for vnstock_data.\n\nRe-exports vnstock's core types for compatibility while adding\ncustom types specific to vnstock_data's broader data coverage.\n"
_U='parquet'
_T='application/octet-stream'
_S='excel'
_R='csv'
_Q='json'
_P='email_notifications'
_O='webhooks'
_N='export_formats'
_M='custom_indicators'
_L='priority_support'
_K='api_access'
_J='advanced_analytics'
_I='historical_data'
_H='real_time_data'
_G='requests_per_day'
_F='requests_per_hour'
_E='requests_per_minute'
_D='data_categories'
_C='data_sources'
_B=False
_A=True
from enum import Enum
from typing import TypedDict,Optional,Any,List
from datetime import datetime
try:from app.lib._vnstock_shared.core.types import ProviderType,MarketType,ExchangeType,TimeFrame
except ImportError:
	class ProviderType(str,Enum):'Stub: Provider types from vnstock.';STOCK='stock';FUTURE='future';OPTION='option'
	class MarketType(str,Enum):'Stub: Market types from vnstock.';STOCK='stock';DERIVATIVE='derivative';FUND='fund'
	class ExchangeType(str,Enum):'Stub: Exchange types from vnstock.';HOSE='hose';HNX='hnx';UPCOM='upcom'
	class TimeFrame(str,Enum):'Stub: Time frames from vnstock.';D1='1D';H4='4H';H1='1H';M15='15m';M5='5m';M1='1m'
class DataCategory(str,Enum):"Data categories supported by vnstock_data (extends vnstock's categories).";QUOTE='quote';COMPANY='company';FINANCIAL='financial';TRADING='trading';LISTING='listing';SCREENER='screener';MACRO='macro';COMMODITY='commodity';MARKET='market';INSIGHT='insight';FUND='fund'
class DataSource(str,Enum):'Data sources available in vnstock_data (more comprehensive than vnstock).';VCI='vci';VND='vnd';MAS='mas';SPL='spl';MBK='mbk';CAFEF='cafef';FMARKET='fmarket';TVS='tvs';VDS='vds'
class SubscriptionTier(str,Enum):'User subscription tier levels for vnstock_data.';GUEST='guest';BASIC='basic';BRONZE='bronze';SILVER='silver';GOLDEN='golden'
class DataQuality(str,Enum):'Data quality levels.';RAW='raw';CLEANED='cleaned';VALIDATED='validated';PREMIUM='premium'
class UpdateFrequency(str,Enum):'Data update frequency.';REAL_TIME='real_time';HOURLY='hourly';DAILY='daily';WEEKLY='weekly';MONTHLY='monthly'
class CacheStrategy(str,Enum):'Cache strategy for data.';NO_CACHE='no_cache';SHORT_TERM='short_term';MEDIUM_TERM='medium_term';LONG_TERM='long_term'
class PremiumQuoteData(TypedDict,total=_B):'Extended quote data for premium users.';symbol:str;price:float;change:float;change_pct:float;volume:int;timestamp:datetime;bid_ask_spread:Optional[float];implied_volatility:Optional[float];options_chain:Optional[dict];technical_indicators:Optional[dict]
class DataSourceInfo(TypedDict):'Information about a data source.';name:str;provider:str;update_frequency:UpdateFrequency;data_quality:DataQuality;subscription_required:bool;tier:SubscriptionTier
class UserProfile(TypedDict):'User profile information.';user_id:str;subscription_tier:SubscriptionTier;api_key:str;rate_limit:int;cache_strategy:CacheStrategy;created_at:datetime;expires_at:Optional[datetime]
class SubscriptionFeatures(TypedDict):'Features available for each subscription tier.';tier:SubscriptionTier;requests_per_minute:int;requests_per_hour:int;requests_per_day:int;data_sources:List[DataSource];data_categories:List[DataCategory];real_time_data:bool;historical_data:bool;advanced_analytics:bool;api_access:bool;priority_support:bool;custom_indicators:bool;export_formats:List[str];webhooks:bool;email_notifications:bool
class RateLimitInfo(TypedDict):'Rate limiting information for user.';tier:SubscriptionTier;current_requests:int;limit_per_minute:int;limit_per_hour:int;limit_per_day:int;reset_time:datetime;remaining_requests:int
class FileTypes:'MIME type mappings.';CSV='text/csv';JSON='application/json';EXCEL='application/vnd.ms-excel';PARQUET=_T;FEATHER=_T
class ParameterNames:'Standardized parameter names.';SYMBOL='symbol';START_DATE='start_date';END_DATE='end_date';INTERVAL='interval';LIMIT='limit';OFFSET='offset';SORT='sort';ORDER='order'
class MethodNames:'Standardized method names.';GET_QUOTE='get_quote';GET_HISTORY='get_history';GET_FINANCIALS='get_financials';GET_INTRADAY='get_intraday';SCREEN='screen'
class SubscriptionConfig:
	'Configuration for subscription tiers.';TIER_LIMITS={SubscriptionTier.GUEST:{_E:15,_F:900,_G:21600},SubscriptionTier.BASIC:{_E:60,_F:3600,_G:86400},SubscriptionTier.BRONZE:{_E:200,_F:12000,_G:288000},SubscriptionTier.SILVER:{_E:500,_F:30000,_G:720000},SubscriptionTier.GOLDEN:{_E:1000,_F:60000,_G:1440000}};TIER_FEATURES={SubscriptionTier.GUEST:{_C:[DataSource.VCI],_D:[DataCategory.QUOTE],_H:_B,_I:_A,_J:_B,_K:_B,_L:_B,_M:_B,_N:[_Q],_O:_B,_P:_B},SubscriptionTier.BASIC:{_C:[DataSource.VCI,DataSource.VND,DataSource.MAS],_D:[DataCategory.QUOTE,DataCategory.COMPANY],_H:_A,_I:_A,_J:_B,_K:_A,_L:_B,_M:_B,_N:[_Q,_R],_O:_B,_P:_B},SubscriptionTier.BRONZE:{_C:[DataSource.VCI,DataSource.VND,DataSource.MAS,DataSource.CAFEF],_D:[DataCategory.QUOTE,DataCategory.COMPANY,DataCategory.FINANCIAL],_H:_A,_I:_A,_J:_A,_K:_A,_L:_A,_M:_B,_N:[_Q,_R,_S],_O:_B,_P:_A},SubscriptionTier.SILVER:{_C:[DataSource.VCI,DataSource.VND,DataSource.MAS,DataSource.CAFEF,DataSource.SPL],_D:[DataCategory.QUOTE,DataCategory.COMPANY,DataCategory.FINANCIAL,DataCategory.TRADING],_H:_A,_I:_A,_J:_A,_K:_A,_L:_A,_M:_A,_N:[_Q,_R,_S,_U],_O:_A,_P:_A},SubscriptionTier.GOLDEN:{_C:list(DataSource),_D:list(DataCategory),_H:_A,_I:_A,_J:_A,_K:_A,_L:_A,_M:_A,_N:[_Q,_R,_S,_U,'feather'],_O:_A,_P:_A}}
	@classmethod
	def get_rate_limit(A,tier):'Get rate limits for a subscription tier.';return A.TIER_LIMITS[tier]
	@classmethod
	def get_features(A,tier):'Get features available for a subscription tier.';return A.TIER_FEATURES[tier]
	@classmethod
	def can_access_source(A,tier,source):'Check if tier can access a data source.';return source in A.TIER_FEATURES[tier][_C]
	@classmethod
	def can_access_category(A,tier,category):'Check if tier can access a data category.';return category in A.TIER_FEATURES[tier][_D]
__all__=['DataCategory','ProviderType','MarketType','ExchangeType','TimeFrame','DataSource','SubscriptionTier','DataQuality','UpdateFrequency','CacheStrategy','PremiumQuoteData','DataSourceInfo','UserProfile','FileTypes','ParameterNames','MethodNames']