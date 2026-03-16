# Reference

- Qualified name: `app.lib.vnstock_data_alt.ui.reference.Reference`
- Signature: `()`

Reference Data Layer (Layer 1).

## Purpose

Reference Data Layer (Layer 1).
Provides access to static/master data for various domains.

## Members

### company

- Kind: `method`
- Signature: `(symbol)`
- Purpose: Access company-specific reference data.

#### Parameters

| Name | Kind | Required | Default | Annotation | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `str` | `VNM`, `TCB` | Ticker symbol (e.g., 'VNM', 'TCB'). |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access company-specific reference data.

### futures

- Kind: `method`
- Signature: `(symbol=None)`
- Purpose: Access index futures reference data (listing or symbol-specific info).

#### Parameters

| Name | Kind | Required | Default | Annotation | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str, optional` | `VN30F2503`, `VN30F1M` | Futures symbol (e.g., 'VN30F2503', 'VN30F1M'). If None, returns listing interface. |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access index futures reference data (listing or symbol-specific info).

Example:
    r = Reference()
    # List all futures indices
    futures_list = r.futures().list()
    
    # Get specific futures info
    futures_info = r.futures('VN30F2503').info()

### warrant

- Kind: `method`
- Signature: `(symbol=None)`
- Purpose: Access covered warrant reference data (info, specifications, pricing).

#### Parameters

| Name | Kind | Required | Default | Annotation | Accepted values | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `symbol` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `str` | `CACB2511`, `CACB25C100` | Warrant symbol (e.g., 'CACB2511', 'CACB25C100'). |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access covered warrant reference data (info, specifications, pricing).

Example:
    r = Reference()
    warrant_info = r.warrant('CACB2511').info()

### industry

- Kind: `property`
- Signature: `(self)`
- Purpose: Access industry reference data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access industry reference data.

### fund

- Kind: `property`
- Signature: `(self)`
- Purpose: Master data for Mutual Funds (Chứng Chỉ Quỹ).

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Master data for Mutual Funds (Chứng Chỉ Quỹ).

### etf

- Kind: `property`
- Signature: `(self)`
- Purpose: Access ETF reference data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access ETF reference data.

### equity

- Kind: `property`
- Signature: `(self)`
- Purpose: Access equity reference data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access equity reference data.

### index

- Kind: `property`
- Signature: `(self)`
- Purpose: Access index reference data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access index reference data.

### bond

- Kind: `property`
- Signature: `(self)`
- Purpose: Access bond reference data.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access bond reference data.

### events

- Kind: `property`
- Signature: `(self)`
- Purpose: Access events reference data (calendar, etc.).

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access events reference data (calendar, etc.).

### search

- Kind: `property`
- Signature: `(self)`
- Purpose: Access global symbol search.

#### Parameters

| Name | Kind | Required | Default | Annotation |
| --- | --- | --- | --- | --- |
| `self` | `POSITIONAL_OR_KEYWORD` | `True` | `None` | `` |

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

Access global symbol search.

### derivatives

- Kind: `method`
- Signature: `()`
- Return type: `DerivativesReference`
- Purpose: [DEPRECATED] Access derivatives reference data.

#### Parameters

_None._

#### Source details

_No source-specific output contract derived._

#### Notes / caveats

[DEPRECATED] Access derivatives reference data.

To fix: Replace `Reference().derivatives().futures(symbol).info()` with `Reference().futures(symbol).info()`

**Examples**
    # Old way (will raise warning):
    r.derivatives().futures('VN30F2503').info()
    
    # New direct way:
    r.futures('VN30F2503').info()
