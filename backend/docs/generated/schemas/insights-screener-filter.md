# insights.screener.filter

- Class: `ScreenerReference`
- Method: `filter`
- Signature: `(limit: int = 2000) -> pd.DataFrame`
- Return type: `pd.DataFrame`
- Normalization mode: `contractual`
- Supported sources: `vci`
- Declared signature: `(limit=2000, **B)`
- Default route source: `vci`
- Default provider: `screener.Screener.filter`

Retrieves full market data (all stocks) with all available criteria (ratios, metrics)

## Purpose

Retrieves full market data (all stocks) with all available criteria (ratios, metrics)
returning a comprehensive DataFrame of the market.
Users can apply advanced filtering logic directly on this DataFrame using Pandas.

## Parameters

| Name | Kind | Required | Default | Annotation | Observed example | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `limit` | `POSITIONAL_OR_KEYWORD` | `False` | `2000` | `int` | `5` | Maximum number of records to retrieve. Defaults to 2000. **kwargs: Additional parameters passed to the provider adapter. |

## Source details

### Source `vci`

#### Raw output contract

- Coverage: `declared`
- Provider: `app.lib.vnstock_data_alt.explorer.vci.screener.Screener`
- Provider method: `filter`

```text
ticker, market_price, daily_price_change_percent, adtv30_days, trading_value_adtv10_days, avg_volume30_days, price_return3_month, price_fluctuation30_days, outperforms_index3_month, rs3_month, ttm_pe, ttm_pb, ttm_roe, npatmi_growth_yoy_qm1, es_volume_vs_avg_volume30_days, en_organ_name, en_organ_short_name, vi_organ_name, vi_organ_short_name, icb_code_lv2, en_sector, icb_code_lv4, ema_time, match_price_time, ema20, price_ema20, ema20_ema50, ema50_ema200, macd_signal_line
```

| Raw | Normalized |
| --- | --- |
| `ticker` | `symbol` |
| `market_price` | `price` |
| `daily_price_change_percent` | `price_change_percent` |
| `adtv30_days` | `avg_value_30d` |
| `trading_value_adtv10_days` | `avg_value_10d` |
| `avg_volume30_days` | `avg_volume_30d` |
| `price_return3_month` | `price_return_3m` |
| `price_fluctuation30_days` | `price_fluctuation_30d` |
| `outperforms_index3_month` | `outperforms_index_3m` |
| `rs3_month` | `rs_3m` |
| `ttm_pe` | `pe` |
| `ttm_pb` | `pb` |
| `ttm_roe` | `roe` |
| `npatmi_growth_yoy_qm1` | `profit_growth_yoy` |
| `es_volume_vs_avg_volume30_days` | `volume_breakout_30d` |
| `en_organ_name` | `company_name_en` |
| `en_organ_short_name` | `short_name_en` |
| `vi_organ_name` | `company_name` |
| `vi_organ_short_name` | `short_name` |
| `icb_code_lv2` | `icb_code2` |
| `en_sector` | `industry_en` |
| `icb_code_lv4` | `icb_code4` |
| `ema_time` | `ema_time` |
| `match_price_time` | `match_price_time` |
| `ema20` | `ema_20` |
| `price_ema20` | `price_ema_20` |
| `ema20_ema50` | `ema_20_ema_50` |
| `ema50_ema200` | `ema_50_ema_200` |
| `macd_signal_line` | `macd_signal` |

#### Normalized output schema

- Coverage: `not-available`

#### Live-observed sample

- Captured at: `2026-03-17T05:26:52.716323+00:00`
- Success: `True`
- Row count: `5`

```text
symbol, exchange, ref_price, ceiling, price, floor, accumulated_value, accumulated_volume, market_cap, price_change_percent, avg_value_30d, avg_value_10d, avg_volume_30d, est_volume, volume_breakout_30d, pe, pb, roe, profit_growth_yoy, revenue_growth_yoy, net_margin, gross_margin, match_price_time, rs_3m, ema_time, ema_20, price_ema_20, ema_20_ema_50, ema_50_ema_200, macd_signal, macd, histogram, outperforms_index_3m, price_return_3m, price_fluctuation_30d, rsi, last_modified_date, company_name_en, short_name_en, company_name, short_name, icb_code2, industry_en, vi_sector, icb_code4, stock_strength, stock_trend, adx, ao, ao_trend
```
- Dtypes: `{'symbol': 'str', 'exchange': 'str', 'ref_price': 'float64', 'ceiling': 'float64', 'price': 'float64', 'floor': 'float64', 'accumulated_value': 'float64', 'accumulated_volume': 'float64', 'market_cap': 'float64', 'price_change_percent': 'float64', 'avg_value_30d': 'float64', 'avg_value_10d': 'float64', 'avg_volume_30d': 'float64', 'est_volume': 'float64', 'volume_breakout_30d': 'float64', 'pe': 'float64', 'pb': 'float64', 'roe': 'float64', 'profit_growth_yoy': 'float64', 'revenue_growth_yoy': 'float64', 'net_margin': 'float64', 'gross_margin': 'float64', 'match_price_time': 'str', 'rs_3m': 'int64', 'ema_time': 'str', 'ema_20': 'float64', 'price_ema_20': 'float64', 'ema_20_ema_50': 'float64', 'ema_50_ema_200': 'float64', 'macd_signal': 'float64', 'macd': 'float64', 'histogram': 'float64', 'outperforms_index_3m': 'float64', 'price_return_3m': 'float64', 'price_fluctuation_30d': 'float64', 'rsi': 'float64', 'last_modified_date': 'str', 'company_name_en': 'str', 'short_name_en': 'str', 'company_name': 'str', 'short_name': 'str', 'icb_code2': 'str', 'industry_en': 'str', 'vi_sector': 'str', 'icb_code4': 'str', 'stock_strength': 'int64', 'stock_trend': 'str', 'adx': 'float64', 'ao': 'float64', 'ao_trend': 'str'}`

```json
[
  {
    "symbol": "APF",
    "exchange": "UPCOM",
    "ref_price": 52000.0,
    "ceiling": 59800.0,
    "price": 52000.0,
    "floor": 44200.0,
    "accumulated_value": 329390000.0,
    "accumulated_volume": 6300.0,
    "market_cap": 1702861628000.0,
    "price_change_percent": 0.0,
    "avg_value_30d": 1119493966.6666667,
    "avg_value_10d": -79.84450105,
    "avg_volume_30d": 24681.5666666667,
    "est_volume": 12858.7627575,
    "volume_breakout_30d": -47.90135111,
    "pe": 10.6679227176,
    "pb": 1.4347934718,
    "roe": 14.11990963,
    "profit_growth_yoy": 106.22716736,
    "revenue_growth_yoy": 81.77275906,
    "net_margin": 2.47179779,
    "gross_margin": 13.9263246,
    "match_price_time": "2026-03-17T04:12:17.94",
    "rs_3m": 95,
    "ema_time": "1773619200",
    "ema_20": 46793.469002432,
    "price_ema_20": 11.12661897,
    "ema_20_ema_50": 6.35881416,
    "ema_50_ema_200": 4.21646054,
    "macd_signal": 1638.3614386299653,
    "macd": 2149.23754044269,
    "histogram": 510.8761018127247,
    "outperforms_index_3m": 30.12354242,
    "price_return_3m": 33.23767552,
    "price_fluctuation_30d": 30.39215686,
    "rsi": 72.60694750441453,
    "last_modified_date": "2026-03-17T04:30:15.185",
    "company_name_en": "Quang Ngai Agricultural Products And Foodstuff Joint Stock Company",
    "short_name_en": "Quang Ngai Agricultural Products",
    "company_name": "Công ty Cổ phần Nông sản Thực phẩm Quảng Ngãi",
    "short_name": "Nông sản Quảng Ngãi",
    "icb_code2": "3500",
    "industry_en": "Food & Beverage",
    "vi_sector": "Thực phẩm và đồ uống",
    "icb_code4": "3577",
    "stock_strength": 93,
    "stock_trend": "STRONG_UPTREND",
    "adx": 37.78687038194197,
    "ao": 4685.2941176471,
    "ao_trend": "ABOVE_ZERO"
  },
  {
    "symbol": "AVG",
    "exchange": "UPCOM",
    "ref_price": 10400.0,
    "ceiling": 11900.0,
    "price": 10400.0,
    "floor": 8900.0,
    "accumulated_value": 347380000.0,
    "accumulated_volume": 33800.0,
    "market_cap": 183871812800.0,
    "price_change_percent": 0.0,
    "avg_value_30d": 1531810846.6666667,
    "avg_value_10d": -86.39796184,
    "avg_volume_30d": 148507.7666666667,
    "est_volume": 64624.10008248,
    "volume_breakout_30d": -56.48436339,
    "pe": 11.4570153003,
    "pb": 0.7698677737,
    "roe": 6.9532344,
    "profit_growth_yoy": 74.62312983,
    "revenue_growth_yoy": 21.59676283,
    "net_margin": 2.40746419,
    "gross_margin": 5.1954244,
    "match_price_time": "2026-03-17T04:21:13.472",
    "rs_3m": 72,
    "ema_time": "1773619200",
    "ema_20": 10314.240599301793,
    "price_ema_20": 0.83146597,
    "ema_20_ema_50": -1.05890405,
    "ema_50_ema_200": -20.72649297,
    "macd_signal": 40.825861494090944,
    "macd": 44.771874579194,
    "histogram": 3.946013085103056,
    "outperforms_index_3m": -0.14383607,
    "price_return_3m": 2.97029703,
    "price_fluctuation_30d": 10.1010101,
    "rsi": 52.25284143379721,
    "last_modified_date": "2026-03-17T04:30:15.185",
    "company_name_en": "Europe Vietnam International Fertilizer Joint Stock Company",
    "short_name_en": "Europe Vietnam International Fertilizer ",
    "company_name": "Công ty Cổ phần Phân Bón Quốc Tế Âu Việt",
    "short_name": "Phân Bón Quốc Tế Âu Việt",
    "icb_code2": "1300",
    "industry_en": "Chemicals",
    "vi_sector": "Hóa chất",
    "icb_code4": "1357",
    "stock_strength": 52,
    "stock_trend": "STRONG_UPTREND",
    "adx": 39.65519525295349,
    "ao": 78.8235294118,
    "ao_trend": "ABOVE_ZERO"
  },
  {
    "symbol": "BTP",
    "exchange": "HOSE",
    "ref_price": 8400.0,
    "ceiling": 8980.0,
    "price": 8430.0,
    "floor": 7820.0,
    "accumulated_value": 72132000.0,
    "accumulated_volume": 8500.0,
    "market_cap": 509893608000.0,
    "price_change_percent": 0.35714286,
    "avg_value_30d": 756158970.3333334,
    "avg_value_10d": -91.27200836,
    "avg_volume_30d": 90769.6333333333,
    "est_volume": 19821.67352469,
    "volume_breakout_30d": -78.16265992,
    "pe": 12.315377885,
    "pb": 0.472552635,
    "roe": 3.85785377,
    "profit_growth_yoy": -28.31999839,
    "revenue_growth_yoy": -8.42389331,
    "net_margin": 15.66986336,
    "gross_margin": 16.26020319,
    "match_price_time": "2026-03-17T04:04:21.131",
    "rs_3m": 30,
    "ema_time": "1773619200",
    "ema_20": 8393.627801521137,
    "price_ema_20": 0.43333109,
    "ema_20_ema_50": -1.99365749,
    "ema_50_ema_200": -12.12192497,
    "macd_signal": -28.284420631682003,
    "macd": -18.53741692587,
    "histogram": 9.747003705812004,
    "outperforms_index_3m": -9.44746643,
    "price_return_3m": -6.33333333,
    "price_fluctuation_30d": 8.91959799,
    "rsi": 50.49637625917011,
    "last_modified_date": "2026-03-17T04:30:15.185",
    "company_name_en": "Ba Ria Thermal Power Joint Stock Company",
    "short_name_en": "Ba Ria Thermal Power",
    "company_name": "Công ty Cổ phần Nhiệt điện Bà Rịa",
    "short_name": "Nhiệt điện Bà Rịa",
    "icb_code2": "7500",
    "industry_en": "Utilities",
    "vi_sector": "Điện, nước & xăng dầu khí đốt",
    "icb_code4": "7535",
    "stock_strength": 36,
    "stock_trend": "STRONG_UPTREND",
    "adx": 28.132489769519257,
    "ao": 119.9117647059,
    "ao_trend": "ABOVE_ZERO"
  }
]
```
