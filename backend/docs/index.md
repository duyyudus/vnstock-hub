# Vendored vnstock Docs

This site documents the vendored `vnstock_alt` and `vnstock_data_alt` packages.

The reference content under `generated/` is produced from source code, schema
registries, compatibility tests, and optional live-observed samples.

The architecture page in this site is scoped to the vendored packages only. It
is not a general architecture guide for the entire backend.

## Start Here

- [Vendored Package Architecture](architecture.md)
- [Overview](generated/index.md)
- [vnstock_alt](generated/packages/vnstock_alt/index.md)
- [vnstock_data_alt](generated/packages/vnstock_data_alt/index.md)
- [Output Schemas](generated/schemas/index.md)
- [Live Samples](generated/live-samples/index.md)
- [Coverage / Limitations](generated/coverage.md)

## Refresh The Docs

From `backend/`:

```bash
uv run python scripts/generate_vnstock_api_docs.py
```

To refresh live-observed samples first:

```bash
uv run python scripts/capture_vnstock_api_samples.py
uv run python scripts/generate_vnstock_api_docs.py
```
