from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.generate_vnstock_api_docs import build_docs_metadata
from scripts.vnstock_api_docs_common import (
    LIVE_ROOT,
    MANIFEST_PATH,
    dummy_value_for_parameter,
    ensure_dir,
    import_from_path,
    read_json,
    serialize_data,
    slugify,
    write_json,
)


AUTO_OPTIONAL_METHOD_PARAMS = {
    "count_back",
    "duration",
    "end",
    "exchange",
    "filter_by",
    "fund_type",
    "get_all",
    "group",
    "interval",
    "lang",
    "limit",
    "page",
    "page_size",
    "period",
    "show_log",
    "start",
    "to_df",
}

AUTO_OPTIONAL_INIT_PARAMS = {
    "code",
    "index",
    "index_symbol",
    "symbol",
    "ticker",
}

SKIPPED_AUTO_PATTERNS = (
    "market.warrant",
    "derivatives.warrant",
    "reference.warrant",
)


def _probe_key(probe: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(probe.get("package", "")),
        str(probe.get("class_path", "")),
        str(probe.get("method", "")),
        str(probe.get("source", "")),
        str(probe.get("schema_key", "")),
    )


def _normalize_source_value(package: str, source: str) -> str:
    if package == "vnstock_alt":
        return source.upper()
    return source.lower()


def _choose_symbol(class_path: str, schema_key: str | None = None) -> str | None:
    context = f"{class_path} {schema_key or ''}".lower()
    if "warrant" in context:
        return None
    if "commodity" in context:
        return "GC=F"
    if "crypto" in context:
        return "BTC"
    if "forex" in context or "exchange_rate" in context:
        return "USDVND"
    if "futures" in context or "future" in context or "derivatives" in context:
        return "VN30F1M"
    if "etf" in context:
        return "E1VFVN30"
    if "fund" in context and "fundamental" not in context:
        return None
    if "index" in context:
        return "VNINDEX"
    return "VCB"


def _sample_value_for_param(
    name: str,
    *,
    class_path: str,
    package: str,
    schema_key: str | None = None,
    source: str | None = None,
) -> Any:
    lowered = name.lower()
    if lowered == "source" and source:
        return _normalize_source_value(package, source)
    if lowered in {"symbol", "ticker", "code", "index_symbol"}:
        return _choose_symbol(class_path, schema_key)
    if lowered == "index":
        return "VNINDEX"
    if lowered in {"start", "from_date"}:
        return "2025-03-01"
    if lowered in {"end", "to_date"}:
        return "2025-03-07"
    if lowered == "interval":
        return "1D"
    if lowered in {"count_back", "limit", "page_size"}:
        return 5
    if lowered == "page":
        return 1
    if lowered == "period":
        return "year"
    if lowered == "duration":
        return "1Y"
    if lowered == "exchange":
        return "HOSE"
    if lowered == "group":
        return "VN30"
    if lowered == "lang":
        return "vi"
    if lowered == "fund_type":
        return "BOND"
    if lowered == "filter_by":
        return "all"
    if lowered == "show_log":
        return False
    if lowered == "to_df":
        return True
    if lowered == "get_all":
        return True
    return dummy_value_for_parameter(name)


def _build_init_kwargs(
    class_path: str,
    *,
    package: str,
    source: str | None = None,
    schema_key: str | None = None,
) -> dict[str, Any] | None:
    class_obj = import_from_path(class_path)
    try:
        signature = inspect.signature(class_obj)
    except (TypeError, ValueError):
        return {}

    kwargs: dict[str, Any] = {}
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY}:
            continue
        should_include = (
            param.default is inspect.Signature.empty
            or param.name == "source"
            or param.name in AUTO_OPTIONAL_INIT_PARAMS
        )
        if not should_include:
            continue
        value = _sample_value_for_param(
            param.name,
            class_path=class_path,
            package=package,
            schema_key=schema_key,
            source=source,
        )
        if value is None and param.default is inspect.Signature.empty:
            return None
        if value is not None:
            kwargs[param.name] = value
    return kwargs


def _build_method_kwargs(
    parameters: list[dict[str, Any]],
    *,
    class_path: str,
    package: str,
    schema_key: str | None = None,
    source: str | None = None,
) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {}
    for param in parameters:
        kind = param.get("kind")
        name = param.get("name")
        if not name or kind in {"VAR_POSITIONAL", "VAR_KEYWORD"}:
            continue
        should_include = param.get("required", False) or name in AUTO_OPTIONAL_METHOD_PARAMS
        if not should_include:
            continue
        value = _sample_value_for_param(
            name,
            class_path=class_path,
            package=package,
            schema_key=schema_key,
            source=source,
        )
        if value is None and param.get("required", False):
            return None
        if value is not None:
            kwargs[name] = value
    return kwargs


def _skip_auto_probe(class_path: str, schema_key: str | None = None) -> bool:
    context = f"{class_path} {schema_key or ''}".lower()
    return any(pattern in context for pattern in SKIPPED_AUTO_PATTERNS)


def _probe_sources(raw_outputs: list[dict[str, Any]], supported_sources: list[str]) -> list[str]:
    explicit_sources = [
        item["source"]
        for item in raw_outputs
        if item.get("source") and item.get("coverage") != "not-available"
    ]
    if explicit_sources:
        return list(dict.fromkeys(explicit_sources))
    return list(dict.fromkeys(source for source in supported_sources if source))


def _auto_schema_probes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for schema in metadata["schemas"]:
        class_path = schema.get("class_path")
        if not class_path or _skip_auto_probe(class_path, schema.get("schema_key")):
            continue
        sources = _probe_sources(schema.get("raw_outputs", []), schema.get("supported_sources") or [])
        for source in sources:
            init_kwargs = _build_init_kwargs(
                class_path,
                package="vnstock_data_alt",
                source=source,
                schema_key=schema.get("schema_key"),
            )
            if init_kwargs is None:
                continue
            method_kwargs = _build_method_kwargs(
                schema.get("parameters", []),
                class_path=class_path,
                package="vnstock_data_alt",
                schema_key=schema.get("schema_key"),
                source=source,
            )
            if method_kwargs is None:
                continue

            probes.append(
                {
                    "package": "vnstock_data_alt",
                    "class_path": class_path,
                    "method": schema["method"],
                    "source": source,
                    "schema_key": schema.get("schema_key"),
                    "init_kwargs": init_kwargs,
                    "method_kwargs": method_kwargs,
                    "probe_origin": "auto-schema",
                }
            )
    return probes


def _auto_package_probes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for package in metadata["packages"]:
        package_name = package["package"]
        for export in package["exports"]:
            if export.get("kind") != "class" or not export.get("supported_sources"):
                continue
            class_path = export["qualified_name"]
            if _skip_auto_probe(class_path):
                continue

            for method in export.get("methods", []):
                if method.get("kind") != "method":
                    continue
                if "backward compatible alias" in method.get("summary", "").lower():
                    continue
                sources = _probe_sources(method.get("raw_outputs", []), export.get("supported_sources", []))
                for source in sources:
                    init_kwargs = _build_init_kwargs(
                        class_path,
                        package=package_name,
                        source=source,
                    )
                    if init_kwargs is None:
                        continue
                    method_kwargs = _build_method_kwargs(
                        method.get("parameters", []),
                        class_path=class_path,
                        package=package_name,
                        source=source,
                    )
                    if method_kwargs is None:
                        continue
                    probes.append(
                        {
                            "package": package_name,
                            "class_path": class_path,
                            "method": method["name"],
                            "source": source,
                            "init_kwargs": init_kwargs,
                            "method_kwargs": method_kwargs,
                            "probe_origin": "auto-package",
                        }
                    )
    return probes


def collect_probe_definitions(
    manifest_path: Path = MANIFEST_PATH,
    *,
    include_auto: bool = True,
) -> list[dict[str, Any]]:
    ordered: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for probe in read_json(manifest_path, default=[]):
        ordered[_probe_key(probe)] = probe

    if include_auto:
        metadata = build_docs_metadata()
        for probe in _auto_schema_probes(metadata) + _auto_package_probes(metadata):
            ordered.setdefault(_probe_key(probe), probe)

    return list(ordered.values())


def run_probe(probe: dict[str, Any]) -> dict[str, Any]:
    class_obj = import_from_path(probe["class_path"])
    instance = class_obj(**probe.get("init_kwargs", {}))
    method = getattr(instance, probe["method"])
    result = method(**probe.get("method_kwargs", {}))

    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "package": probe["package"],
        "class_name": class_obj.__name__,
        "class_path": probe["class_path"],
        "method": probe["method"],
        "source": probe.get("source"),
        "schema_key": probe.get("schema_key"),
        "probe_origin": probe.get("probe_origin", "manifest"),
        "init_kwargs": probe.get("init_kwargs", {}),
        "method_kwargs": probe.get("method_kwargs", {}),
        "success": True,
        "row_count": None,
        "columns": [],
        "dtypes": {},
        "preview_rows": [],
        "result_type": type(result).__name__,
        "error": None,
    }

    if isinstance(result, pd.DataFrame):
        payload["row_count"] = int(len(result.index))
        payload["columns"] = list(result.columns)
        payload["dtypes"] = {column: str(dtype) for column, dtype in result.dtypes.items()}
        payload["preview_rows"] = serialize_data(result.head(3).to_dict(orient="records"))
    else:
        payload["preview_rows"] = serialize_data(result)
    return payload


def capture_samples(
    manifest_path: Path = MANIFEST_PATH,
    live_root: Path = LIVE_ROOT,
    *,
    include_auto: bool = True,
    clean_existing: bool = True,
) -> list[dict[str, Any]]:
    probes = collect_probe_definitions(manifest_path=manifest_path, include_auto=include_auto)
    ensure_dir(live_root)
    results: list[dict[str, Any]] = []

    if clean_existing:
        for path in live_root.glob("*.json"):
            path.unlink()

    for probe in probes:
        try:
            payload = run_probe(probe)
        except Exception as exc:
            payload = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "package": probe.get("package"),
                "class_name": probe.get("class_path", "").rsplit(".", 1)[-1],
                "class_path": probe.get("class_path"),
                "method": probe.get("method"),
                "source": probe.get("source"),
                "schema_key": probe.get("schema_key"),
                "probe_origin": probe.get("probe_origin", "manifest"),
                "init_kwargs": probe.get("init_kwargs", {}),
                "method_kwargs": probe.get("method_kwargs", {}),
                "success": False,
                "row_count": 0,
                "columns": [],
                "dtypes": {},
                "preview_rows": [],
                "result_type": None,
                "error": repr(exc),
            }

        results.append(payload)
        file_name = "__".join(
            [
                slugify(str(payload.get("package", ""))),
                slugify(str(payload.get("class_name", ""))),
                slugify(str(payload.get("method", ""))),
                slugify(str(payload.get("source", ""))),
            ]
        ).strip("_")
        write_json(live_root / f"{file_name}.json", payload)

    write_json(live_root / "index.json", results)
    return results


def validate_snapshot(payload: dict[str, Any]) -> None:
    required_keys = {
        "captured_at",
        "package",
        "class_name",
        "class_path",
        "method",
        "source",
        "init_kwargs",
        "method_kwargs",
        "success",
        "row_count",
        "columns",
        "dtypes",
        "preview_rows",
        "error",
    }
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture live sample outputs for vendored vnstock docs.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--live-root", type=Path, default=LIVE_ROOT)
    parser.add_argument("--manifest-only", action="store_true", help="Capture only explicitly listed probes.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not remove older captured snapshots before writing new ones.")
    args = parser.parse_args()

    results = capture_samples(
        manifest_path=args.manifest,
        live_root=args.live_root,
        include_auto=not args.manifest_only,
        clean_existing=not args.keep_existing,
    )
    print(json.dumps({"captured": len(results), "live_root": str(args.live_root)}, indent=2))


if __name__ == "__main__":
    main()
