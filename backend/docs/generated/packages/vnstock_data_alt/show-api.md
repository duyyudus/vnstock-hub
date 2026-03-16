# show_api

- Qualified name: `app.lib.vnstock_data_alt.ui.helper.show_api`
- Signature: `(layer=None, show_navigation=True)`
- Purpose: Displays a visual API Tree of available endpoints.

## Parameters

| Name | Kind | Required | Default | Annotation | Description |
| --- | --- | --- | --- | --- | --- |
| `layer` | `POSITIONAL_OR_KEYWORD` | `False` | `None` | `` | (Optional) Limit display to a specific Layer (e.g., Market(), 'Market'). If empty (None), displays all 6 library layers. |
| `show_navigation` | `POSITIONAL_OR_KEYWORD` | `False` | `True` | `` | If True, displays intermediate navigation methods (returning domain objects). Default is True. |

## Notes / caveats

Displays a visual API Tree of available endpoints.
Only shows endpoint methods (returning data), hiding backward compatible aliases.
