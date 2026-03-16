from __future__ import annotations

import argparse
import inspect
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.vnstock_api_docs_common import (
    GENERATED_ROOT,
    LIVE_ROOT,
    clean_docstring,
    candidate_column_lists,
    derive_docstring_columns,
    derive_method_columns,
    ensure_dir,
    format_annotation,
    import_from_path,
    import_module,
    load_package_modules,
    maybe_getattr,
    parse_docstring_sections,
    read_live_snapshots,
    safe_default,
    safe_instantiate,
    sanitize_docstring_for_markdown,
    serialize_data,
    signature_parameters,
    signature_to_string,
    slugify,
    summarize_docstring,
    write_json,
)


PACKAGE_NAMES = {
    "vnstock_alt": "app.lib.vnstock_alt",
    "vnstock_data_alt": "app.lib.vnstock_data_alt",
}

PACKAGE_INDEX_GROUPS = {
    "vnstock_alt": [
        ("Core Entry Point", {"Vnstock"}),
        ("Market Data Adapters", {"Quote", "Listing", "Company", "Finance", "Trading", "Fund"}),
        ("Constants", {"INDICES_INFO", "INDICES_MAP", "INDEX_GROUPS", "SECTOR_IDS", "EXCHANGES"}),
        ("Support Modules", {"connector"}),
    ],
    "vnstock_data_alt": [
        ("Provider Adapters", {"Quote", "Company", "Finance", "Listing", "Trading", "CommodityPrice", "TopStock", "Fund"}),
        ("Unified UI Namespaces", {"Reference", "Market", "Insights", "Fundamental", "Macro", "Analytics"}),
        ("Helpers", {"show_api", "show_doc"}),
    ],
}


def _callable_signature(value: Any, drop_first: bool = False) -> str:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return "()"
    params = list(signature.parameters.values())
    if drop_first and params:
        params = params[1:]
    return str(signature.replace(parameters=params))


def _callable_parameters(value: Any, drop_first: bool = False) -> list[dict[str, Any]]:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return []
    params = list(signature.parameters.values())
    if drop_first and params:
        params = params[1:]
    return [
        {
            "name": param.name,
            "kind": param.kind.name,
            "annotation": format_annotation(param.annotation),
            "default": safe_default(param.default),
            "required": param.default is inspect.Signature.empty,
        }
        for param in params
    ]


def _route_callable(route: tuple[str, str, str, str] | None, env: dict[str, Any]) -> Any | None:
    if not route:
        return None
    source, provider_type, _class_name, provider_method = route
    try:
        provider_class = env["data_registry"].get(provider_type, source)
    except Exception:
        return None
    return getattr(provider_class, provider_method, None)


def _enrich_from_docstring(value: Any, parameters: list[dict[str, Any]], return_type: str | None) -> tuple[list[dict[str, Any]], str | None]:
    parsed = parse_docstring_sections(value)
    doc_params = parsed["params"]
    doc_return_type = parsed["return_type"]

    if (not parameters or all(param["kind"] == "VAR_KEYWORD" for param in parameters)) and doc_params:
        parameters = [
            {
                "name": item["name"],
                "kind": "DOC",
                "annotation": item["annotation"],
                "default": None,
                "required": False,
                "description": item["description"],
                "example": item.get("example"),
                "accepted_values": item.get("accepted_values", []),
            }
            for item in doc_params
        ]
    elif parameters and doc_params:
        doc_param_map = {item["name"]: item for item in doc_params}
        enriched_parameters: list[dict[str, Any]] = []
        for param in parameters:
            doc_item = doc_param_map.get(param["name"])
            if not doc_item:
                enriched_parameters.append(param)
                continue
            enriched_param = dict(param)
            if doc_item.get("annotation") and not enriched_param.get("annotation"):
                enriched_param["annotation"] = doc_item["annotation"]
            if doc_item.get("description"):
                enriched_param["description"] = doc_item["description"]
            if doc_item.get("example"):
                enriched_param["example"] = doc_item["example"]
            if doc_item.get("accepted_values"):
                enriched_param["accepted_values"] = doc_item["accepted_values"]
            enriched_parameters.append(enriched_param)
        parameters = enriched_parameters

    if not return_type and doc_return_type:
        return_type = doc_return_type

    return parameters, return_type


def _is_unhelpful_signature(signature: str) -> bool:
    normalized = signature.replace(" ", "")
    if normalized in {"()", "(**B)", "(*A,**B)", "(*args,**kwargs)", "(*args:Any,**kwargs:Any)->Any"}:
        return True
    if re.search(r"\*[A-Z]", normalized) or re.search(r"\*\*[A-Z]", normalized):
        return True
    return False


def _format_signature_param(param: dict[str, Any]) -> str:
    name = param["name"]
    annotation = param.get("annotation")
    default = param.get("default")
    kind = param.get("kind")

    prefix = ""
    if kind == "VAR_POSITIONAL":
        prefix = "*"
    elif kind == "VAR_KEYWORD":
        prefix = "**"

    text = f"{prefix}{name}"
    if annotation:
        text += f": {annotation}"
    if kind not in {"VAR_POSITIONAL", "VAR_KEYWORD"} and not param.get("required", True):
        text += f" = {repr(default)}"
    return text


def _effective_signature(
    declared_signature: str,
    parameters: list[dict[str, Any]],
    return_type: str | None,
) -> str:
    if not _is_unhelpful_signature(declared_signature):
        return declared_signature
    if not parameters:
        return declared_signature
    body = ", ".join(_format_signature_param(param) for param in parameters)
    if return_type:
        return f"({body}) -> {return_type}"
    return f"({body})"


def _parameters_unhelpful(parameters: list[dict[str, Any]]) -> bool:
    if not parameters:
        return True
    if all(param["kind"] in {"VAR_POSITIONAL", "VAR_KEYWORD"} for param in parameters):
        return True
    if all(len(param["name"]) == 1 for param in parameters):
        return True
    return False


def _drop_placeholder_variadics(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(parameters) <= 1:
        return parameters

    useful_params = [
        param
        for param in parameters
        if not (
            param.get("kind") in {"VAR_POSITIONAL", "VAR_KEYWORD"}
            and (len(param.get("name", "")) == 1 or param.get("name") in {"args", "kwargs"})
        )
    ]
    return useful_params or parameters


def _format_observed_value(value: Any) -> str:
    return repr(value) if not isinstance(value, str) else value


def _infer_observed_value_from_aliases(param: dict[str, Any], sample: dict[str, Any]) -> str | None:
    param_name = param.get("name")
    init_kwargs = sample.get("init_kwargs", {}) or {}
    method_kwargs = sample.get("method_kwargs", {}) or {}

    if param_name == "symbols_list" and "symbol" in init_kwargs:
        return repr([init_kwargs["symbol"]])
    if param_name == "page_size" and "limit" in method_kwargs:
        return _format_observed_value(method_kwargs["limit"])
    return None


def _fallback_observed_example(param: dict[str, Any]) -> str:
    if not param.get("required", True):
        default = param.get("default")
        if default not in (None, "", [], {}):
            return f"omitted; default {default!r}"
    return "omitted in live probe"


def _enrich_parameters_from_live_samples(
    parameters: list[dict[str, Any]],
    live_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not parameters or not live_samples:
        return parameters

    observed_values: dict[str, str] = {}
    for sample in live_samples:
        init_kwargs = sample.get("init_kwargs", {}) or {}
        for name, value in init_kwargs.items():
            if name not in observed_values:
                observed_values[name] = _format_observed_value(value)
        method_kwargs = sample.get("method_kwargs", {}) or {}
        for name, value in method_kwargs.items():
            if name not in observed_values:
                observed_values[name] = _format_observed_value(value)

    enriched: list[dict[str, Any]] = []
    for param in parameters:
        updated = dict(param)
        if updated["name"] in observed_values:
            updated["observed_example"] = observed_values[updated["name"]]
        else:
            for sample in live_samples:
                inferred = _infer_observed_value_from_aliases(updated, sample)
                if inferred is not None:
                    updated["observed_example"] = inferred
                    break
            if "observed_example" not in updated:
                updated["observed_example"] = _fallback_observed_example(updated)
        enriched.append(updated)
    return enriched


def _provider_signature_hint_for_class_method(
    cls: type,
    method_name: str,
    package_key: str,
    env: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    for source in _class_supported_sources(cls, package_key, env):
        provider_class = _provider_class_for_source(cls, package_key, source, env)
        if provider_class is None or not hasattr(provider_class, method_name):
            continue
        provider_callable = getattr(provider_class, method_name)
        parameters, return_type = _enrich_from_docstring(
            provider_callable,
            _callable_parameters(provider_callable, drop_first=True),
            format_annotation(getattr(inspect.signature(provider_callable), "return_annotation", inspect.Signature.empty)),
        )
        if not _parameters_unhelpful(parameters):
            return parameters, return_type, source
    return [], None, None


def _load_environment() -> dict[str, Any]:
    load_package_modules("app.lib.vnstock_alt.explorer")
    load_package_modules("app.lib.vnstock_alt.connector")
    load_package_modules("app.lib.vnstock_data_alt.explorer")
    load_package_modules("app.lib.vnstock_data_alt.ui.domains")
    import_module("app.lib.vnstock_data_alt.ui.schemas")

    compat = import_module("tests.test_alt_package_compat")
    shared_registry = import_module("app.lib._vnstock_shared.core.registry").ProviderRegistry
    data_registry = import_module("app.lib.vnstock_data_alt.core.registry").ProviderRegistry
    schema_core = import_module("app.lib.vnstock_data_alt.ui.schemas.core")
    ui_config = import_module("app.lib.vnstock_data_alt.ui.config")

    return {
        "compat": compat,
        "shared_registry": shared_registry,
        "data_registry": data_registry,
        "schema_core": schema_core,
        "ui_config": ui_config,
    }


def _registry_for_package(package_key: str, env: dict[str, Any]) -> Any | None:
    if package_key == "vnstock_alt":
        return env["shared_registry"]
    if package_key == "vnstock_data_alt":
        return env["data_registry"]
    return None


def _public_class_members(cls: type, preferred_names: list[str] | None = None) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    if preferred_names is not None:
        for name in preferred_names:
            member = getattr(cls, name, None)
            if member is None:
                continue
            members.append(
                {
                    "name": name,
                    "kind": "method",
                    "callable": member,
                }
            )
        return members

    for name, member in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            members.append({"name": name, "kind": "property", "callable": member.fget})
            continue
        if inspect.isfunction(member) or inspect.ismethoddescriptor(member):
            members.append({"name": name, "kind": "method", "callable": member})
    return members


def _class_supported_sources(cls: type, package_key: str, env: dict[str, Any]) -> list[str]:
    registry = _registry_for_package(package_key, env)
    if hasattr(cls, "_module_name") and registry is not None:
        return registry.list_available(getattr(cls, "_module_name"))

    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return []

    source_param = signature.parameters.get("source")
    if source_param and source_param.default not in (inspect.Signature.empty, None):
        default = source_param.default
        if isinstance(default, str):
            return [default.lower()]
    return []


def _provider_class_for_source(
    cls: type,
    package_key: str,
    source: str,
    env: dict[str, Any],
) -> Any | None:
    registry = _registry_for_package(package_key, env)
    if registry is None or not hasattr(cls, "_module_name"):
        return None

    try:
        return registry.get(getattr(cls, "_module_name"), source)
    except Exception:
        return None


def _derive_provider_outputs(
    cls: type,
    method_name: str,
    package_key: str,
    env: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for source in _class_supported_sources(cls, package_key, env):
        provider_class = _provider_class_for_source(cls, package_key, source, env)
        if provider_class is None or not hasattr(provider_class, method_name):
            outputs.append(
                {
                    "source": source,
                    "coverage": "not-available",
                    "raw_columns": [],
                    "notes": [f"Provider does not expose `{method_name}` for this source."],
                }
            )
            continue

        provider_module = import_module(provider_class.__module__)
        candidates = candidate_column_lists(provider_module, method_name)
        if candidates:
            const_name, raw_columns = candidates[0]
            coverage = "declared"
            notes = [f"Derived from `{provider_class.__module__}.{const_name}`."]
        else:
            provider_method = getattr(provider_class, method_name)
            raw_columns = derive_method_columns(provider_method, provider_module)
            if raw_columns:
                coverage = "declared"
                notes = ["Derived from static analysis of provider DataFrame shaping logic."]
            else:
                raw_columns = derive_docstring_columns(provider_method) or []
                if raw_columns:
                    coverage = "partially-derived"
                    notes = ["Derived from provider docstring column hints."]
                else:
                    coverage = "partially-derived"
                    notes = ["No explicit column constant or recoverable DataFrame-shaping pattern found in provider method."]

        outputs.append(
            {
                "source": source,
                "coverage": coverage,
                "provider_class": f"{provider_class.__module__}.{provider_class.__name__}",
                "provider_method": method_name,
                "raw_columns": raw_columns,
                "notes": notes,
            }
        )
    return outputs


def _route_catalog() -> dict[str, tuple[str, str, str, str]]:
    registry_module = import_module("app.lib.vnstock_data_alt.ui._registry")
    route_maps = [
        registry_module.REFERENCE_SOURCES,
        registry_module.MARKET_SOURCES,
        registry_module.FUNDAMENTAL_SOURCES,
        registry_module.INSIGHTS_SOURCES,
        registry_module.MACRO_SOURCES,
    ]
    flattened: dict[str, tuple[str, str, str, str]] = {}
    for route_map in route_maps:
        for domain_name, methods in route_map.items():
            for method_name, route in methods.items():
                flattened[f"{domain_name}.{method_name}"] = route
    return flattened


def _endpoint_strictness(bound_method: Any) -> str:
    try:
        source = inspect.getsource(bound_method)
    except (OSError, TypeError):
        return "contractual"
    if "strict=False" in source:
        return "best-effort"
    return "contractual"


def _collect_ui_endpoint_lookup(env: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    route_catalog = _route_catalog()
    for module_name in load_package_modules("app.lib.vnstock_data_alt.ui.domains"):
        module = import_module(module_name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            instance = safe_instantiate(cls)
            if instance is None or not hasattr(instance, "_domain_name") or not hasattr(instance, "_sources_config"):
                continue

            domain_name = instance._domain_name
            for name, unbound_member in inspect.getmembers(cls):
                if name.startswith("_"):
                    continue
                if not (inspect.isfunction(unbound_member) or inspect.ismethoddescriptor(unbound_member)):
                    continue
                key = f"{domain_name}.{name}"
                route = route_catalog.get(key)
                if key not in env["schema_core"]._STANDARD_COLS_REGISTRY and route is None:
                    continue
                bound = getattr(instance, name)
                parameters, return_type = _enrich_from_docstring(
                    bound,
                    signature_parameters(bound),
                    format_annotation(getattr(inspect.signature(bound), "return_annotation", inspect.Signature.empty)),
                )
                parameters = _drop_placeholder_variadics(parameters)
                provider_callable = _route_callable(route, env)
                if provider_callable and (not parameters or all(param["kind"] == "VAR_KEYWORD" for param in parameters)):
                    parameters, hinted_return_type = _enrich_from_docstring(
                        provider_callable,
                        _callable_parameters(provider_callable, drop_first=True),
                        format_annotation(getattr(inspect.signature(provider_callable), "return_annotation", inspect.Signature.empty)),
                    )
                    if hinted_return_type and not return_type:
                        return_type = hinted_return_type
                parameters = _drop_placeholder_variadics(parameters)
                declared_signature = signature_to_string(bound)
                lookup[key] = {
                    "schema_key": key,
                    "class_name": cls.__name__,
                    "class_path": f"{cls.__module__}.{cls.__name__}",
                    "method": name,
                    "signature": _effective_signature(declared_signature, parameters, return_type),
                    "declared_signature": declared_signature,
                    "parameters": parameters,
                    "return_type": return_type,
                    "summary": summarize_docstring(bound),
                    "docstring": clean_docstring(bound),
                    "strictness": _endpoint_strictness(bound),
                    "default_route": route,
                }
    return lookup


def _derive_ui_raw_outputs(
    schema_key: str,
    endpoint_meta: dict[str, Any] | None,
    schema_map: dict[str, dict[str, str]],
    standard_columns: list[str],
    env: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    default_route = endpoint_meta.get("default_route") if endpoint_meta else None
    default_route_source = default_route[0] if default_route else None
    default_route_method = default_route[3] if default_route else schema_key.rsplit(".", 1)[-1]
    route_provider_type = default_route[1] if default_route else None
    route_class_name = default_route[2] if default_route else None

    sources = sorted(set(schema_map) | ({default_route_source} if default_route_source else set()))
    for source in sources:
        mapping = schema_map.get(source, {})
        raw_columns = list(mapping.keys()) if mapping else []
        coverage = "declared" if mapping else "partially-derived"
        notes: list[str] = []

        provider_path = None
        if source == default_route_source and route_provider_type and route_class_name:
            registry = env["data_registry"]
            try:
                provider_class = registry.get(route_provider_type, source)
                provider_path = f"{provider_class.__module__}.{provider_class.__name__}"
                provider_module = import_module(provider_class.__module__)
                candidates = candidate_column_lists(provider_module, default_route_method)
                if candidates:
                    const_name, candidate_columns = candidates[0]
                    raw_columns = candidate_columns
                    coverage = "declared"
                    notes.append(f"Derived from `{provider_class.__module__}.{const_name}`.")
            except Exception:
                notes.append(f"Default route points to `{route_class_name}.{default_route_method}` but provider lookup failed.")

        if not raw_columns and standard_columns:
            raw_columns = list(standard_columns)
            notes.append("No source-specific raw mapping declared; using normalized columns as a best-effort approximation.")

        outputs.append(
            {
                "source": source,
                "coverage": coverage if raw_columns else "not-available",
                "provider_class": provider_path,
                "provider_method": default_route_method if source == default_route_source else None,
                "raw_columns": raw_columns,
                "raw_to_normalized_map": mapping,
                "notes": notes,
            }
        )
    return outputs


def _collect_schema_catalog(
    env: dict[str, Any],
    live_snapshots: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    endpoint_lookup = _collect_ui_endpoint_lookup(env)
    schema_core = env["schema_core"]
    schema_keys = sorted(set(schema_core._STANDARD_COLS_REGISTRY) | set(schema_core._SCHEMA_REGISTRY))
    schemas: list[dict[str, Any]] = []
    live_by_schema_key = {
        sample["schema_key"]: []
        for samples in live_snapshots.values()
        for sample in samples
        if sample.get("schema_key")
    }
    for samples in live_snapshots.values():
        for sample in samples:
            schema_key = sample.get("schema_key")
            if schema_key:
                live_by_schema_key.setdefault(schema_key, []).append(sample)

    for schema_key in schema_keys:
        endpoint_meta = endpoint_lookup.get(schema_key)
        standard_columns = list(schema_core._STANDARD_COLS_REGISTRY.get(schema_key, []))
        schema_map = schema_core._SCHEMA_REGISTRY.get(schema_key, {})
        relevant_enum_map = {
            column: mapping
            for column, mapping in schema_core._ENUM_REGISTRY.items()
            if column in standard_columns
            or any(column in source_map.values() for source_map in schema_map.values())
        }
        raw_outputs = _derive_ui_raw_outputs(schema_key, endpoint_meta, schema_map, standard_columns, env)
        default_route = endpoint_meta.get("default_route") if endpoint_meta else None
        class_name = endpoint_meta["class_name"] if endpoint_meta else ""
        method_name = endpoint_meta["method"] if endpoint_meta else schema_key.rsplit(".", 1)[-1]
        schema_live_samples = live_by_schema_key.get(schema_key, live_snapshots.get(("vnstock_data_alt", class_name, method_name), []))
        schema_parameters = _enrich_parameters_from_live_samples(
            endpoint_meta.get("parameters", []) if endpoint_meta else [],
            schema_live_samples,
        )

        schemas.append(
            {
                "schema_key": schema_key,
                "summary": endpoint_meta.get("summary", "") if endpoint_meta else "",
                "docstring": endpoint_meta.get("docstring", "") if endpoint_meta else "",
                "signature": endpoint_meta.get("signature", "()") if endpoint_meta else "()",
                "declared_signature": endpoint_meta.get("declared_signature", endpoint_meta.get("signature", "()")) if endpoint_meta else "()",
                "parameters": schema_parameters,
                "return_type": endpoint_meta.get("return_type", "pd.DataFrame") if endpoint_meta else "pd.DataFrame",
                "class_name": class_name,
                "class_path": endpoint_meta.get("class_path", "") if endpoint_meta else "",
                "method": method_name,
                "strictness": endpoint_meta.get("strictness", "contractual") if endpoint_meta else "contractual",
                "default_route": {
                    "source": default_route[0],
                    "provider_type": default_route[1],
                    "class_name": default_route[2],
                    "provider_method": default_route[3],
                }
                if default_route
                else None,
                "supported_sources": sorted({item["source"] for item in raw_outputs}),
                "raw_outputs": raw_outputs,
                "normalized_output": {
                    "columns": standard_columns,
                    "enum_map": relevant_enum_map,
                    "coverage": "declared" if standard_columns else "not-available",
                },
                "live_samples": schema_live_samples,
            }
        )
    return schemas


def _collect_package_exports(
    package_key: str,
    env: dict[str, Any],
    live_snapshots: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    compat = env["compat"]
    module = import_module(PACKAGE_NAMES[package_key])
    contract_map = {}
    if package_key == "vnstock_alt":
        contract_map = compat.VNSTOCK_ROOT_METHODS
    elif package_key == "vnstock_data_alt":
        contract_map = compat.VNSTOCK_DATA_ROOT_METHODS

    exports: list[dict[str, Any]] = []
    for name in getattr(module, "__all__", []):
        obj = maybe_getattr(module, name)
        if obj is None:
            continue

        if inspect.isclass(obj):
            members = _public_class_members(obj, contract_map.get(name))
            class_entry = {
                "name": name,
                "kind": "class",
                "qualified_name": f"{obj.__module__}.{obj.__name__}",
                "signature": signature_to_string(obj),
                "summary": summarize_docstring(obj),
                "docstring": clean_docstring(obj),
                "supported_sources": _class_supported_sources(obj, package_key, env),
                "methods": [],
            }
            for member in members:
                callable_obj = member["callable"]
                method_name = member["name"]
                parameters, return_type = _enrich_from_docstring(
                    callable_obj,
                    _callable_parameters(callable_obj, drop_first=member["kind"] == "method"),
                    format_annotation(getattr(inspect.signature(callable_obj), "return_annotation", inspect.Signature.empty))
                    if member["kind"] == "method"
                    else None,
                )
                parameters = _drop_placeholder_variadics(parameters)
                declared_signature = _callable_signature(callable_obj, drop_first=member["kind"] == "method")
                signature_hint_source = None
                if member["kind"] == "method" and (_is_unhelpful_signature(declared_signature) or _parameters_unhelpful(parameters)):
                    hinted_parameters, hinted_return_type, signature_hint_source = _provider_signature_hint_for_class_method(
                        obj,
                        method_name,
                        package_key,
                        env,
                    )
                    if hinted_parameters:
                        parameters = hinted_parameters
                    if hinted_return_type and not return_type:
                        return_type = hinted_return_type
                parameters = _drop_placeholder_variadics(parameters)
                method_live_samples = live_snapshots.get((package_key, name, method_name), [])
                parameters = _enrich_parameters_from_live_samples(parameters, method_live_samples)
                method_entry = {
                    "name": method_name,
                    "kind": member["kind"],
                    "signature": _effective_signature(declared_signature, parameters, return_type),
                    "declared_signature": declared_signature,
                    "signature_hint_source": signature_hint_source,
                    "parameters": parameters,
                    "return_type": return_type,
                    "summary": summarize_docstring(callable_obj),
                    "docstring": clean_docstring(callable_obj),
                    "supported_sources": class_entry["supported_sources"],
                    "raw_outputs": _derive_provider_outputs(obj, method_name, package_key, env)
                    if member["kind"] == "method" and class_entry["supported_sources"]
                    else [],
                    "normalized_output": None,
                    "live_samples": method_live_samples,
                }
                class_entry["methods"].append(method_entry)
            exports.append(class_entry)
            continue

        if callable(obj):
            parameters, return_type = _enrich_from_docstring(
                obj,
                signature_parameters(obj),
                format_annotation(getattr(inspect.signature(obj), "return_annotation", inspect.Signature.empty)),
            )
            parameters = _drop_placeholder_variadics(parameters)
            declared_signature = signature_to_string(obj)
            exports.append(
                {
                    "name": name,
                    "kind": "function",
                    "qualified_name": f"{obj.__module__}.{obj.__name__}",
                    "signature": _effective_signature(declared_signature, parameters, return_type),
                    "declared_signature": declared_signature,
                    "summary": summarize_docstring(obj),
                    "docstring": clean_docstring(obj),
                    "parameters": parameters,
                    "return_type": return_type,
                    "live_samples": live_snapshots.get((package_key, name, name), []),
                }
            )
            continue

        exports.append(
            {
                "name": name,
                "kind": "value",
                "qualified_name": f"{module.__name__}.{name}",
                "value": safe_default(obj),
            }
        )

    return {
        "package": package_key,
        "module": module.__name__,
        "exports": exports,
    }


def build_docs_metadata(live_root: Path = LIVE_ROOT) -> dict[str, Any]:
    env = _load_environment()
    live_snapshots_all = read_live_snapshots(live_root)
    live_snapshots = {
        key: [sample for sample in samples if sample.get("success")]
        for key, samples in live_snapshots_all.items()
    }
    packages = [
        _collect_package_exports("vnstock_alt", env, live_snapshots),
        _collect_package_exports("vnstock_data_alt", env, live_snapshots),
    ]
    schemas = _collect_schema_catalog(env, live_snapshots)
    live_index = _build_live_index(live_root)
    return {
        "packages": packages,
        "schemas": schemas,
        "live_samples": live_index,
    }


def _build_live_index(live_root: Path) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for path in sorted(live_root.glob("*.json")) if live_root.exists() else []:
        if path.name == "index.json":
            continue
        payload = import_module("json").loads(path.read_text())
        index.append(payload)
    return index


def _render_parameters(parameters: list[dict[str, Any]]) -> str:
    if not parameters:
        return "_None._"
    has_description = any(param.get("description") for param in parameters)
    has_examples = any(param.get("example") for param in parameters)
    has_observed_examples = any(param.get("observed_example") for param in parameters)
    has_accepted_values = any(param.get("accepted_values") for param in parameters)
    if has_description or has_examples or has_observed_examples or has_accepted_values:
        headers = ["Name", "Kind", "Required", "Default", "Annotation"]
        if has_examples:
            headers.append("Example")
        if has_observed_examples:
            headers.append("Observed example")
        if has_accepted_values:
            headers.append("Accepted values")
        if has_description:
            headers.append("Description")
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    else:
        lines = ["| Name | Kind | Required | Default | Annotation |", "| --- | --- | --- | --- | --- |"]
    for param in parameters:
        if has_description or has_examples or has_observed_examples or has_accepted_values:
            row = [
                f"`{param['name']}`",
                f"`{param['kind']}`",
                f"`{param['required']}`",
                f"`{param['default']}`",
                f"`{param['annotation'] or ''}`",
            ]
            if has_examples:
                example = param.get("example")
                row.append(f"`{example}`" if example else "")
            if has_observed_examples:
                observed_example = param.get("observed_example")
                row.append(f"`{observed_example}`" if observed_example else "")
            if has_accepted_values:
                accepted_values = param.get("accepted_values") or []
                row.append(", ".join(f"`{value}`" for value in accepted_values))
            if has_description:
                row.append(param.get("description", ""))
            lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append(
                f"| `{param['name']}` | `{param['kind']}` | `{param['required']}` | "
                f"`{param['default']}` | `{param['annotation'] or ''}` |"
            )
    return "\n".join(lines)


def _render_single_raw_output(item: dict[str, Any]) -> str:
    parts: list[str] = [
        f"- Coverage: `{item['coverage']}`",
    ]
    if item.get("provider_class"):
        parts.append(f"- Provider: `{item['provider_class']}`")
    if item.get("provider_method"):
        parts.append(f"- Provider method: `{item['provider_method']}`")
    if item.get("raw_columns"):
        parts.extend(
            [
                "",
                "```text",
                ", ".join(item["raw_columns"]),
                "```",
            ]
        )
    else:
        parts.append("")
        parts.append("_No raw columns derived for this source._")
    if item.get("raw_to_normalized_map"):
        parts.extend(
            [
                "",
                "| Raw | Normalized |",
                "| --- | --- |",
            ]
        )
        for raw_name, normalized_name in item["raw_to_normalized_map"].items():
            parts.append(f"| `{raw_name}` | `{normalized_name}` |")
    for note in item.get("notes", []):
        parts.append(f"- Note: {note}")
    return "\n".join(parts).strip()


def _render_normalized_output(normalized: dict[str, Any] | None) -> str:
    if not normalized:
        return "_No normalized schema declared for this API surface._"
    lines = [f"- Coverage: `{normalized['coverage']}`"]
    columns = normalized.get("columns", [])
    if columns:
        lines.append("")
        lines.append("```text")
        lines.append(", ".join(columns))
        lines.append("```")
    enum_map = normalized.get("enum_map", {})
    if enum_map:
        lines.append("")
        lines.append("Enum/value normalization:")
        lines.append("")
        for column, mapping in enum_map.items():
            lines.append(f"- `{column}`: {mapping}")
    return "\n".join(lines)


def _render_live_samples(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return (
            "_No live sample is attached to this exact endpoint yet._\n\n"
            "Live samples come from both explicit probes in "
            "`backend/docs/live_probe_manifest.json` and auto-generated per-source probes. "
            "If a source still has no sample here, that source either failed during capture or "
            "is not currently probeable with the default inputs."
        )
    parts: list[str] = []
    for sample in samples:
        parts.append(f"- Captured at: `{sample.get('captured_at', 'unknown date')}`")
        parts.append(f"- Success: `{sample.get('success')}`")
        if sample.get("error"):
            parts.append(f"- Error: `{sample['error']}`")
        parts.append(f"- Row count: `{sample.get('row_count')}`")
        if sample.get("columns"):
            parts.append("")
            parts.append("```text")
            parts.append(", ".join(str(column) for column in sample["columns"]))
            parts.append("```")
        if sample.get("dtypes"):
            parts.append(f"- Dtypes: `{sample['dtypes']}`")
        if sample.get("preview_rows"):
            parts.append("")
            parts.append("```json")
            parts.append(import_module("json").dumps(sample["preview_rows"], indent=2, ensure_ascii=False))
            parts.append("```")
        parts.append("")
    return "\n".join(parts).strip()


def _render_source_sections(
    raw_outputs: list[dict[str, Any]],
    normalized: dict[str, Any] | None,
    live_samples: list[dict[str, Any]],
    *,
    heading_level: int,
) -> str:
    if not raw_outputs:
        parts = ["_No source-specific output contract derived._"]
        if normalized:
            parts.extend(
                [
                    "",
                    "#" * (heading_level + 1) + " Normalized output schema",
                    "",
                    _render_normalized_output(normalized),
                ]
            )
        if live_samples:
            parts.extend(
                [
                    "",
                    "#" * (heading_level + 1) + " Live-observed sample",
                    "",
                    _render_live_samples(live_samples),
                ]
            )
        return "\n".join(parts)

    live_by_source: dict[str, list[dict[str, Any]]] = {}
    for sample in live_samples:
        live_by_source.setdefault(sample.get("source") or "", []).append(sample)

    parts: list[str] = []
    for item in raw_outputs:
        source = item.get("source", "unknown")
        parts.extend(
            [
                f"{'#' * heading_level} Source `{source}`",
                "",
                f"{'#' * (heading_level + 1)} Raw output contract",
                "",
                _render_single_raw_output(item),
                "",
                f"{'#' * (heading_level + 1)} Normalized output schema",
                "",
                _render_normalized_output(normalized),
                "",
                f"{'#' * (heading_level + 1)} Live-observed sample",
                "",
                _render_live_samples(live_by_source.get(source, [])),
                "",
            ]
        )
    return "\n".join(parts).strip()


def _write_package_docs(package: dict[str, Any], output_root: Path) -> None:
    package_dir = ensure_dir(output_root / "packages" / package["package"])
    index_lines = [
        f"# {package['package']}",
        "",
        f"Module: `{package['module']}`",
        "",
    ]
    grouped_names = set()
    for heading, names in PACKAGE_INDEX_GROUPS.get(package["package"], []):
        visible = [export for export in package["exports"] if export["name"] in names]
        if not visible:
            continue
        index_lines.extend([f"## {heading}", ""])
        for export in visible:
            grouped_names.add(export["name"])
            if export["kind"] in {"class", "function"}:
                filename = f"{slugify(export['name'])}.md"
                index_lines.append(f"- [{export['name']}]({filename})")
            else:
                index_lines.append(f"- `{export['name']}`")
        index_lines.append("")

    leftovers = [export for export in package["exports"] if export["name"] not in grouped_names]
    if leftovers:
        index_lines.extend(["## Other Exports", ""])
        for export in leftovers:
            if export["kind"] in {"class", "function"}:
                filename = f"{slugify(export['name'])}.md"
                index_lines.append(f"- [{export['name']}]({filename})")
            else:
                index_lines.append(f"- `{export['name']}`")
    (package_dir / "index.md").write_text("\n".join(index_lines) + "\n")

    for export in package["exports"]:
        if export["kind"] == "class":
            lines = [
                f"# {export['name']}",
                "",
                f"- Qualified name: `{export['qualified_name']}`",
                f"- Signature: `{export['signature']}`",
            ]
            if export.get("declared_signature") and export["declared_signature"] != export["signature"]:
                lines.append(f"- Declared signature: `{export['declared_signature']}`")
            if export["supported_sources"]:
                lines.append(f"- Supported sources: `{', '.join(export['supported_sources'])}`")
            if export["summary"]:
                lines.extend(["", export["summary"]])
            if export["docstring"]:
                lines.extend(["", "## Purpose", "", sanitize_docstring_for_markdown(export["docstring"])])

            lines.extend(["", "## Members", ""])
            for method in export["methods"]:
                lines.extend(
                    [
                        f"### {method['name']}",
                        "",
                        f"- Kind: `{method['kind']}`",
                        f"- Signature: `{method['signature']}`",
                    ]
                )
                if method.get("declared_signature") and method["declared_signature"] != method["signature"]:
                    lines.append(f"- Declared signature: `{method['declared_signature']}`")
                if method.get("signature_hint_source"):
                    lines.append(f"- Effective signature source: provider `{method['signature_hint_source']}`")
                if method["return_type"]:
                    lines.append(f"- Return type: `{method['return_type']}`")
                if method["summary"]:
                    lines.append(f"- Purpose: {method['summary']}")
                lines.extend(
                    [
                        "",
                        "#### Parameters",
                        "",
                        _render_parameters(method["parameters"]),
                        "",
                        "#### Source details",
                        "",
                        _render_source_sections(
                            method["raw_outputs"],
                            method["normalized_output"],
                            method["live_samples"],
                            heading_level=5,
                        ),
                        "",
                    ]
                )
                if method["docstring"]:
                    lines.extend(["#### Notes / caveats", "", sanitize_docstring_for_markdown(method["docstring"]), ""])
            (package_dir / f"{slugify(export['name'])}.md").write_text("\n".join(lines).strip() + "\n")
            continue

        if export["kind"] == "function":
            lines = [
                f"# {export['name']}",
                "",
                f"- Qualified name: `{export['qualified_name']}`",
                f"- Signature: `{export['signature']}`",
            ]
            if export.get("declared_signature") and export["declared_signature"] != export["signature"]:
                lines.append(f"- Declared signature: `{export['declared_signature']}`")
            if export.get("return_type"):
                lines.append(f"- Return type: `{export['return_type']}`")
            if export.get("summary"):
                lines.append(f"- Purpose: {export['summary']}")
            lines.extend(["", "## Parameters", "", _render_parameters(export["parameters"])])
            if export.get("docstring"):
                lines.extend(["", "## Notes / caveats", "", sanitize_docstring_for_markdown(export["docstring"])])
            (package_dir / f"{slugify(export['name'])}.md").write_text("\n".join(lines).strip() + "\n")


def _write_schema_docs(schemas: list[dict[str, Any]], output_root: Path) -> None:
    schema_dir = ensure_dir(output_root / "schemas")
    schema_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for schema in schemas:
        parts = schema["schema_key"].split(".")
        top_level = parts[0]
        subgroup = ".".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) == 2 else "")
        schema_groups.setdefault(top_level, {}).setdefault(subgroup, []).append(schema)

    index_lines = [
        "# Output Schemas",
        "",
        "Schema keys are grouped by domain so related endpoints stay together.",
        "",
    ]
    for top_level in sorted(schema_groups):
        index_lines.extend([f"## {top_level}", ""])
        subgroup_map = schema_groups[top_level]
        subgroup_names = sorted(subgroup_map)
        for subgroup in subgroup_names:
            entries = sorted(subgroup_map[subgroup], key=lambda item: item["schema_key"])
            if subgroup:
                index_lines.extend([f"### {subgroup}", ""])
            for schema in entries:
                filename = f"{slugify(schema['schema_key'])}.md"
                index_lines.append(f"- [{schema['schema_key']}]({filename})")
            index_lines.append("")
    (schema_dir / "index.md").write_text("\n".join(index_lines) + "\n")

    for schema in schemas:
        lines = [
            f"# {schema['schema_key']}",
            "",
            f"- Class: `{schema['class_name']}`" if schema["class_name"] else "- Class: `_unknown_`",
            f"- Method: `{schema['method']}`",
            f"- Signature: `{schema['signature']}`",
            f"- Return type: `{schema['return_type']}`",
            f"- Normalization mode: `{schema['strictness']}`",
            f"- Supported sources: `{', '.join(schema['supported_sources'])}`",
        ]
        if schema.get("declared_signature") and schema["declared_signature"] != schema["signature"]:
            lines.append(f"- Declared signature: `{schema['declared_signature']}`")
        if schema["default_route"]:
            lines.extend(
                [
                    f"- Default route source: `{schema['default_route']['source']}`",
                    f"- Default provider: `{schema['default_route']['provider_type']}.{schema['default_route']['class_name']}.{schema['default_route']['provider_method']}`",
                ]
            )
        if schema["summary"]:
            lines.extend(["", schema["summary"]])
        if schema["docstring"]:
            lines.extend(["", "## Purpose", "", sanitize_docstring_for_markdown(schema["docstring"])])

        lines.extend(
            [
                "",
                "## Parameters",
                "",
                _render_parameters(schema["parameters"]),
                "",
                "## Source details",
                "",
                _render_source_sections(
                    schema["raw_outputs"],
                    schema["normalized_output"],
                    schema["live_samples"],
                    heading_level=3,
                ),
            ]
        )
        (schema_dir / f"{slugify(schema['schema_key'])}.md").write_text("\n".join(lines).strip() + "\n")


def _write_overview(metadata: dict[str, Any], output_root: Path) -> None:
    package_counts = {package["package"]: len(package["exports"]) for package in metadata["packages"]}
    lines = [
        "# Vendored vnstock API Docs",
        "",
        "This docs set is generated from the vendored source code and local compatibility contracts.",
        "",
        "## Packages",
        "",
    ]
    for package_name, count in package_counts.items():
        lines.append(f"- `{package_name}`: {count} exported surfaces")
    lines.extend(
        [
            f"- `schema keys`: {len(metadata['schemas'])}",
            f"- `live snapshots`: {len(metadata['live_samples'])}",
            "",
            "## Browse",
            "",
            "- [vnstock_alt](packages/vnstock_alt/index.md)",
            "- [vnstock_data_alt](packages/vnstock_data_alt/index.md)",
            "- [Output Schemas](schemas/index.md)",
            "- [Live Samples](live-samples/index.md)",
            "- [Coverage / Limitations](coverage.md)",
        ]
    )
    (output_root / "index.md").write_text("\n".join(lines) + "\n")


def _write_coverage(metadata: dict[str, Any], output_root: Path) -> None:
    coverage = Counter()
    for package in metadata["packages"]:
        for export in package["exports"]:
            for method in export.get("methods", []):
                for item in method.get("raw_outputs", []):
                    coverage[item["coverage"]] += 1
    for schema in metadata["schemas"]:
        for item in schema["raw_outputs"]:
            coverage[item["coverage"]] += 1

    lines = [
        "# Coverage / Limitations",
        "",
        "Coverage levels reflect how confidently the docs can describe output shape from source:",
        "",
        "- `declared`: derived from explicit schema maps or provider column constants",
        "- `partially-derived`: source gives useful clues but not a complete column contract",
        "- `live-observed`: observed in a captured sample snapshot",
        "- `not-available`: no source-derived or live-observed output contract was found",
        "",
        "## Counts",
        "",
    ]
    for key in ["declared", "partially-derived", "live-observed", "not-available"]:
        lines.append(f"- `{key}`: {coverage.get(key, 0)}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `vnstock_data_alt` UI/domain layers have richer normalized-schema coverage because they ship explicit schema registries.",
            "- `vnstock_alt` output shape is often only partially derivable unless provider modules publish clear column constants.",
            "- Live snapshots are evidence, not the contract. The source remains the primary contract.",
        ]
    )
    (output_root / "coverage.md").write_text("\n".join(lines) + "\n")


def _write_live_samples(metadata: dict[str, Any], output_root: Path) -> None:
    live_dir = ensure_dir(output_root / "live-samples")
    lines = ["# Live Samples", ""]
    if not metadata["live_samples"]:
        lines.append("No live snapshots have been captured yet.")
    else:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for sample in metadata["live_samples"]:
            grouped.setdefault(sample.get("package", "unknown"), {}).setdefault(sample.get("class_name", "unknown"), []).append(sample)
        for package_name in sorted(grouped):
            lines.extend([f"## {package_name}", ""])
            for class_name in sorted(grouped[package_name]):
                lines.extend([f"### {class_name}", ""])
                for sample in sorted(grouped[package_name][class_name], key=lambda item: (item.get("method", ""), item.get("source", ""))):
                    lines.extend(
                        [
                            f"#### {sample.get('method')}",
                            "",
                            f"- Source: `{sample.get('source')}`",
                            f"- Captured at: `{sample.get('captured_at')}`",
                            f"- Success: `{sample.get('success')}`",
                            f"- Row count: `{sample.get('row_count')}`",
                            "",
                        ]
                    )
    (live_dir / "index.md").write_text("\n".join(lines) + "\n")


def render_docs(metadata: dict[str, Any], output_root: Path = GENERATED_ROOT) -> None:
    ensure_dir(output_root)
    ensure_dir(output_root / "packages")
    ensure_dir(output_root / "schemas")
    ensure_dir(output_root / "live-samples")

    _write_overview(metadata, output_root)
    _write_coverage(metadata, output_root)
    _write_live_samples(metadata, output_root)
    for package in metadata["packages"]:
        _write_package_docs(package, output_root)
    _write_schema_docs(metadata["schemas"], output_root)
    write_json(output_root / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local API docs for vendored vnstock packages.")
    parser.add_argument("--output-root", type=Path, default=GENERATED_ROOT)
    args = parser.parse_args()

    metadata = build_docs_metadata()
    render_docs(metadata, output_root=args.output_root)
    print(f"Generated docs at {args.output_root}")


if __name__ == "__main__":
    main()
