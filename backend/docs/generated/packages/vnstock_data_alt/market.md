# Market

- Qualified name: `app.lib.vnstock_data_alt.ui.market.Market`
- Signature: `(index='VNINDEX', random_agent=False, show_log=False, **B)`

Market Data Layer (Layer 2).

## Purpose

Market Data Layer (Layer 2).
Provides access to real-time and historical pricing data across all asset classes.

## Members

### pe

- Kind: `method`
- Signature: `(duration='5Y')`

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `duration` | `POSITIONAL_OR_KEYWORD` | `False` | `5Y` | `` |

#### Source details

_No source-specific output contract derived._

### pb

- Kind: `method`
- Signature: `(duration='5Y')`

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `duration` | `POSITIONAL_OR_KEYWORD` | `False` | `5Y` | `` |

#### Source details

_No source-specific output contract derived._

### evaluation

- Kind: `method`
- Signature: `(duration='5Y')`

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `duration` | `POSITIONAL_OR_KEYWORD` | `False` | `5Y` | `` |

#### Source details

_No source-specific output contract derived._

### equity

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access equity market data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access equity market data.

### index

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access index market data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access index market data.

### futures

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access futures market data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access futures market data.

### warrant

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access warrant market data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access warrant market data.

### etf

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access ETF market data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access ETF market data.

### fund

- Kind: `method`
- Signature: `(symbol=None)`
- Purpose: Access historical NAVs and portfolio compositions for a specific Mutual Fund.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access historical NAVs and portfolio compositions for a specific Mutual Fund.

### crypto

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access crypto market data (e.g., 'BTC').

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access crypto market data (e.g., 'BTC').

### forex

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access forex market data (e.g., 'USDVND').

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access forex market data (e.g., 'USDVND').

### commodity

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access commodity market data (e.g., 'GC=F').

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access commodity market data (e.g., 'GC=F').

### quote

- Kind: `method`
- Signature: `(symbols_list)`
- Declared signature: `(symbols_list, **A)`
- Purpose: Real-time multi-symbol pricing snapshot.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Real-time multi-symbol pricing snapshot.

### price_board

- Kind: `method`
- Signature: `(symbols_list)`
- Declared signature: `(symbols_list, **B)`
- Purpose: [Backward Compatible Alias] Relays to `.quote()`.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `symbols_list` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[Backward Compatible Alias] Relays to `.quote()`.
Provides real-time multi-symbol pricing snapshot.
