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

- Captured at: `2026-03-16T11:15:06.140554+00:00`
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
    "ref_price": 48000.0,
    "ceiling": 55200.0,
    "price": 53200.0,
    "floor": 40800.0,
    "accumulated_value": 3670320000.0,
    "accumulated_volume": 70600.0,
    "market_cap": 1742158434800.0,
    "price_change_percent": 10.83333333,
    "avg_value_30d": 1005500983.3333334,
    "avg_value_10d": 172.10849857,
    "avg_volume_30d": 22543.9333333333,
    "est_volume": 71034.09726132,
    "volume_breakout_30d": 215.09185292,
    "pe": 9.8524433693,
    "pb": 1.3251147202,
    "roe": 14.11990963,
    "profit_growth_yoy": 106.22716736,
    "revenue_growth_yoy": 81.77275906,
    "net_margin": 2.47179779,
    "gross_margin": 13.9263246,
    "match_price_time": "2026-03-16T07:58:21.478",
    "rs_3m": 93,
    "ema_time": "1773360000",
    "ema_20": 46245.41310790063,
    "price_ema_20": 15.03843608,
    "ema_20_ema_50": 5.8994926,
    "ema_50_ema_200": 3.68409196,
    "macd_signal": 1510.6424130712662,
    "macd": 1930.32322909252,
    "histogram": 419.6808160212538,
    "outperforms_index_3m": 34.12723661,
    "price_return_3m": 34.96276347,
    "price_fluctuation_30d": 30.39215686,
    "rsi": 79.13913335967256,
    "last_modified_date": "2026-03-16T08:24:08.958",
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
    "adx": 37.07174406515456,
    "ao": 3788.2336764706,
    "ao_trend": "ABOVE_ZERO"
  },
  {
    "symbol": "ARM",
    "exchange": "HNX",
    "ref_price": 46700.0,
    "ceiling": 51300.0,
    "price": 46800.0,
    "floor": 42100.0,
    "accumulated_value": 4680000.0,
    "accumulated_volume": 100.0,
    "market_cap": 145608044400.0,
    "price_change_percent": 0.21413276,
    "avg_value_30d": 18165726.666666668,
    "avg_value_10d": 2661.0619469,
    "avg_volume_30d": 413.3,
    "est_volume": 106.24999932,
    "volume_breakout_30d": -74.2922818,
    "pe": 24.4962124585,
    "pb": 3.5786548003,
    "roe": 14.98200222,
    "profit_growth_yoy": 26.67458536,
    "revenue_growth_yoy": -33.99235475,
    "net_margin": 2.01652329,
    "gross_margin": 13.52849838,
    "match_price_time": "2026-03-16T07:30:00.139",
    "rs_3m": 99,
    "ema_time": "1772150400",
    "ema_20": 37937.47421347244,
    "price_ema_20": 23.36087462,
    "ema_20_ema_50": 18.50688538,
    "ema_50_ema_200": 14.15896622,
    "macd_signal": 4313.565940716838,
    "macd": 5554.239248796119,
    "histogram": 1240.673308079281,
    "outperforms_index_3m": 84.14471029,
    "price_return_3m": 84.98023715,
    "price_fluctuation_30d": 84.98023715,
    "rsi": 98.64296546413439,
    "last_modified_date": "2026-03-16T08:24:08.976",
    "company_name_en": "General Aviation Import Export Joint Stock Company",
    "short_name_en": "General Aviation Import Export",
    "company_name": "Công ty Cổ phần Xuất nhập khẩu Hàng không",
    "short_name": "XNK Hàng không",
    "icb_code2": "2700",
    "industry_en": "Industrial Goods & Services",
    "vi_sector": "Hàng & Dịch vụ Công nghiệp",
    "icb_code4": "2797",
    "stock_strength": 88,
    "stock_trend": "STRONG_UPTREND",
    "adx": 79.09465345867685,
    "ao": 15358.6294117647,
    "ao_trend": "ABOVE_ZERO"
  },
  {
    "symbol": "AVG",
    "exchange": "UPCOM",
    "ref_price": 10500.0,
    "ceiling": 12000.0,
    "price": 10500.0,
    "floor": 9000.0,
    "accumulated_value": 1715480000.0,
    "accumulated_volume": 164300.0,
    "market_cap": 185639811000.0,
    "price_change_percent": 0.0,
    "avg_value_30d": 1504690583.3333333,
    "avg_value_10d": -39.21557599,
    "avg_volume_30d": 146047.7666666667,
    "est_volume": 152968.96551672,
    "volume_breakout_30d": 4.7389967,
    "pe": 11.5305350805,
    "pb": 0.7748080228,
    "roe": 6.9532344,
    "profit_growth_yoy": 74.62312983,
    "revenue_growth_yoy": 21.59676283,
    "net_margin": 2.40746419,
    "gross_margin": 5.1954244,
    "match_price_time": "2026-03-16T08:20:00.273",
    "rs_3m": 76,
    "ema_time": "1773360000",
    "ema_20": 10305.213293960927,
    "price_ema_20": 1.89017636,
    "ema_20_ema_50": -1.15503118,
    "ema_50_ema_200": -20.88513944,
    "macd_signal": 39.83935821984718,
    "macd": 42.730711821742,
    "histogram": 2.89135360189482,
    "outperforms_index_3m": 3.12486918,
    "price_return_3m": 3.96039604,
    "price_fluctuation_30d": 10.1010101,
    "rsi": 54.51377503753946,
    "last_modified_date": "2026-03-16T08:24:09",
    "company_name_en": "Europe Vietnam International Fertilizer Joint Stock Company",
    "short_name_en": "Europe Vietnam International Fertilizer ",
    "company_name": "Công ty Cổ phần Phân Bón Quốc Tế Âu Việt",
    "short_name": "Phân Bón Quốc Tế Âu Việt",
    "icb_code2": "1300",
    "industry_en": "Chemicals",
    "vi_sector": "Hóa chất",
    "icb_code4": "1357",
    "stock_strength": 58,
    "stock_trend": "STRONG_UPTREND",
    "adx": 38.72486366724153,
    "ao": 37.6470588235,
    "ao_trend": "ABOVE_ZERO"
  }
]
```
