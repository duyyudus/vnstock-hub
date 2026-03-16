from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import re
import sys
import ast
import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

DOCS_ROOT = BACKEND_ROOT / "docs"
GENERATED_ROOT = DOCS_ROOT / "generated"
LIVE_ROOT = GENERATED_ROOT / "live"
MANIFEST_PATH = DOCS_ROOT / "live_probe_manifest.json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def safe_default(value: Any) -> Any:
    if value is inspect.Signature.empty:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return value
    return repr(value)


def format_annotation(annotation: Any) -> str | None:
    if annotation is inspect.Signature.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    if getattr(annotation, "__module__", "") == "builtins":
        return getattr(annotation, "__name__", repr(annotation))
    text = repr(annotation)
    return text.replace("typing.", "").replace("pandas.core.frame.", "")


def clean_docstring(value: Any) -> str:
    return inspect.cleandoc(inspect.getdoc(value) or "")


def sanitize_docstring_for_markdown(doc: str) -> str:
    if not doc:
        return ""

    skip_sections = {
        "args",
        "arguments",
        "parameters",
        "returns",
        "tham số",
        "trả về",
    }
    lines = doc.splitlines()
    result: list[str] = []
    i = 0
    skipping = False

    def is_named_section(line: str) -> bool:
        normalized = line.strip().rstrip(":").strip().lower()
        return normalized in skip_sections

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if is_named_section(line):
            skipping = True
            i += 1
            if i < len(lines) and re.fullmatch(r"[-=]{3,}", lines[i].strip()):
                i += 1
            continue

        if skipping:
            if not stripped:
                i += 1
                continue

            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            looks_like_new_section = stripped.endswith(":") or bool(re.fullmatch(r"[-=]{3,}", next_line))
            if looks_like_new_section:
                skipping = False
            else:
                i += 1
                continue

        if i + 1 < len(lines) and re.fullmatch(r"[-=]{3,}", lines[i + 1].strip()):
            result.append(f"**{stripped.rstrip(':')}**")
            i += 2
            continue

        if stripped.endswith(":") and stripped.rstrip(":").strip().lower() in {"examples", "ví dụ", "notes", "ghi chú"}:
            result.append(f"**{stripped.rstrip(':')}**")
            i += 1
            continue

        result.append(line)
        i += 1

    return "\n".join(result).strip()


def summarize_docstring(value: Any) -> str:
    doc = clean_docstring(value)
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def _docstring_example_values(doc: str) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped.startswith(">>>"):
            continue
        for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,)\n]+)", stripped):
            name = match.group(1)
            value = match.group(2).strip()
            examples.setdefault(name, [])
            if value not in examples[name]:
                examples[name].append(value)
    return examples


def _param_hints_from_description(description: str) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    if not description:
        return hints

    format_match = re.search(r"(?:format|định dạng)\s+([A-Z0-9:\- ]+(?:\s+or\s+[A-Z0-9:\- ]+)*)", description, flags=re.IGNORECASE)
    if format_match:
        hints["example"] = format_match.group(1).strip()

    quoted_values = re.findall(r"'([^']+)'", description)
    if len(quoted_values) >= 2:
        hints["accepted_values"] = quoted_values
        return hints

    paren_groups = re.findall(r"\(([^()]+)\)", description)
    for group in paren_groups:
        if "," not in group:
            continue
        parts = [part.strip(" .'\"") for part in group.split(",")]
        parts = [part for part in parts if part and len(part) <= 24]
        if len(parts) >= 2 and all(" " not in part or part.upper() == part for part in parts):
            hints["accepted_values"] = parts
            break
    return hints


def parse_docstring_sections(value: Any) -> dict[str, Any]:
    doc = clean_docstring(value)
    if not doc:
        return {"params": [], "return_type": None}

    lines = doc.splitlines()
    example_values = _docstring_example_values(doc)
    params: list[dict[str, Any]] = []
    return_type: str | None = None
    section: str | None = None
    current_param: dict[str, Any] | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped in {"Args:", "Arguments:"}:
            section = "args"
            current_param = None
            continue
        if stripped == "Returns:":
            section = "returns"
            current_param = None
            continue
        if (
            stripped.endswith(":")
            and stripped not in {"Args:", "Arguments:", "Returns:"}
            and raw_line == stripped
        ):
            section = None
            current_param = None
            continue

        if section == "args":
            match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(?:\(([^)]+)\))?:\s*(.*)$", raw_line)
            if match:
                current_param = {
                    "name": match.group(1),
                    "annotation": match.group(2),
                    "description": match.group(3).strip(),
                }
                params.append(current_param)
                continue
            if current_param and stripped:
                current_param["description"] = f"{current_param['description']} {stripped}".strip()
            continue

        if section == "returns":
            if not stripped:
                continue
            match = re.match(r"^\s*([a-zA-Z0-9_.\[\] |]+)\s*:\s*(.*)$", raw_line)
            if match:
                return_type = match.group(1).strip()
            elif return_type is None:
                return_type = stripped
            continue

    return {
        "params": [
            (
                lambda hints: {
                    **param,
                    **hints,
                    "example": example_values.get(param["name"], [None])[0] or hints.get("example"),
                }
            )(_param_hints_from_description(param["description"]))
            for param in params
        ],
        "return_type": return_type,
    }


def signature_to_string(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "()"


def signature_parameters(value: Any) -> list[dict[str, Any]]:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return []

    params: list[dict[str, Any]] = []
    for param in signature.parameters.values():
        params.append(
            {
                "name": param.name,
                "kind": param.kind.name,
                "annotation": format_annotation(param.annotation),
                "default": safe_default(param.default),
                "required": param.default is inspect.Signature.empty,
            }
        )
    return params


def serialize_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): serialize_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_data(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(serialize_data(payload), indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def import_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def load_package_modules(package_name: str) -> list[str]:
    imported: list[str] = []
    package = import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if not package_paths:
        return imported

    for package_path in package_paths:
        base_path = Path(package_path)
        if not base_path.exists():
            continue
        for file_path in sorted(base_path.rglob("*.py")):
            if file_path.name.startswith("_"):
                continue
            relative = file_path.relative_to(base_path)
            if file_path.name == "__init__.py":
                module_suffix = ".".join(relative.parent.parts)
            else:
                module_suffix = ".".join(relative.with_suffix("").parts)
            module_name = package_name if not module_suffix else f"{package_name}.{module_suffix}"
            if module_name in imported:
                continue
            try:
                import_module(module_name)
                imported.append(module_name)
            except Exception:
                continue
    return imported


def safe_instantiate(cls: type) -> Any | None:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return None

    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.default is not inspect.Signature.empty:
            continue

        value = dummy_value_for_parameter(param.name)
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional.append(value)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = value

    try:
        return cls(*positional, **kwargs)
    except Exception:
        return None


def dummy_value_for_parameter(name: str) -> Any:
    name = name.lower()
    if "symbol" in name:
        return "VCB"
    if "index" in name:
        return "VNINDEX"
    if "source" in name:
        return "kbs"
    if "exchange" in name:
        return "HOSE"
    if name in {"board"}:
        return "stock"
    if name in {"duration"}:
        return "5Y"
    if name in {"period"}:
        return "year"
    if name in {"limit", "page_size", "page", "count_back"}:
        return 1
    return "sample"


def maybe_getattr(module: Any, name: str) -> Any | None:
    try:
        return getattr(module, name)
    except AttributeError:
        return None


def candidate_column_lists(module: Any, method_name: str) -> list[tuple[str, list[str]]]:
    tokens = {
        method_name.upper(),
        method_name.replace("_", "").upper(),
    }
    score_items: list[tuple[int, str, list[str]]] = []
    for const_name, value in vars(module).items():
        if not isinstance(value, (list, tuple)):
            continue
        if not value or not all(isinstance(item, str) for item in value):
            continue
        upper_name = const_name.upper()
        score = 0
        for token in tokens:
            if token and token in upper_name:
                score += 10
        if "STANDARD_COLUMNS" in upper_name:
            score += 5
        elif upper_name.endswith("_COLUMNS"):
            score += 3
        if score > 0:
            score_items.append((score, const_name, list(value)))
    score_items.sort(key=lambda item: (-item[0], item[1]))
    return [(name, columns) for _, name, columns in score_items]


def read_live_snapshots(live_root: Path = LIVE_ROOT) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    snapshots: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    if not live_root.exists():
        return snapshots

    for path in sorted(live_root.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = read_json(path, default={})
        key = (
            payload.get("package", ""),
            payload.get("class_name", ""),
            payload.get("method", ""),
        )
        snapshots.setdefault(key, []).append(payload)
    return snapshots


def import_from_path(dotted_path: str) -> Any:
    module_name, attr_name = dotted_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def _resolve_scalar_expr(expr: ast.AST, env: dict[str, Any], local_lists: dict[str, list[str]]) -> Any:
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.Name):
        if expr.id in local_lists:
            return local_lists[expr.id]
        return env.get(expr.id)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        value = _resolve_scalar_expr(expr.operand, env, local_lists)
        if isinstance(value, (int, float)):
            return -value
    return None


def _resolve_list_expr(
    expr: ast.AST,
    env: dict[str, Any],
    local_lists: dict[str, list[str]],
    current_columns: list[str] | None = None,
) -> list[str] | None:
    if isinstance(expr, (ast.List, ast.Tuple)):
        items: list[str] = []
        for elt in expr.elts:
            value = _resolve_scalar_expr(elt, env, local_lists)
            if isinstance(value, str):
                items.append(value)
            else:
                nested = _resolve_list_expr(elt, env, local_lists, current_columns=current_columns)
                if nested is None:
                    return None
                items.extend(nested)
        return items

    if isinstance(expr, ast.Name):
        if expr.id in local_lists:
            return list(local_lists[expr.id])
        value = env.get(expr.id)
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return list(value)
        return None

    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _resolve_list_expr(expr.left, env, local_lists, current_columns=current_columns)
        right = _resolve_list_expr(expr.right, env, local_lists, current_columns=current_columns)
        if left is not None and right is not None:
            return left + right
        return None

    if isinstance(expr, ast.ListComp) and len(expr.generators) == 1:
        generator = expr.generators[0]
        iter_list = _resolve_list_expr(generator.iter, env, local_lists, current_columns=current_columns)
        if iter_list is None and isinstance(generator.iter, ast.Attribute):
            if (
                isinstance(generator.iter.value, ast.Name)
                and generator.iter.attr == "columns"
                and current_columns is not None
            ):
                iter_list = list(current_columns)
        if iter_list is None:
            return None

        if isinstance(expr.elt, ast.Name) and isinstance(generator.target, ast.Name) and expr.elt.id == generator.target.id:
            return iter_list

        if (
            isinstance(expr.elt, ast.Call)
            and isinstance(expr.elt.func, ast.Attribute)
            and isinstance(expr.elt.func.value, ast.Name)
            and isinstance(generator.target, ast.Name)
            and expr.elt.func.value.id == generator.target.id
            and expr.elt.func.attr == "replace"
            and len(expr.elt.args) >= 2
        ):
            old = _resolve_scalar_expr(expr.elt.args[0], env, local_lists)
            new = _resolve_scalar_expr(expr.elt.args[1], env, local_lists)
            if isinstance(old, str) and isinstance(new, str):
                return [item.replace(old, new) for item in iter_list]

        if (
            isinstance(expr.elt, ast.Call)
            and isinstance(expr.elt.func, ast.Name)
            and expr.elt.func.id == "camel_to_snake"
            and len(expr.elt.args) == 1
            and isinstance(expr.elt.args[0], ast.Name)
            and isinstance(generator.target, ast.Name)
            and expr.elt.args[0].id == generator.target.id
        ):
            return [re.sub(r"(?<!^)(?=[A-Z])", "_", item).lower().replace("__", "_") for item in iter_list]

    return None


def _resolve_dict_expr(expr: ast.AST, env: dict[str, Any], local_lists: dict[str, list[str]]) -> dict[str, str] | None:
    if not isinstance(expr, ast.Dict):
        return None
    result: dict[str, str] = {}
    for key_node, value_node in zip(expr.keys, expr.values):
        key = _resolve_scalar_expr(key_node, env, local_lists)
        value = _resolve_scalar_expr(value_node, env, local_lists)
        if not isinstance(key, str) or not isinstance(value, str):
            return None
        result[key] = value
    return result


def _apply_dataframe_call(
    call: ast.Call,
    current_columns: list[str] | None,
    env: dict[str, Any],
    local_lists: dict[str, list[str]],
) -> list[str] | None:
    if isinstance(call.func, ast.Name):
        if call.func.id in {"reorder_cols", "filter_columns_by_language"} and call.args:
            return current_columns
        return current_columns

    if not isinstance(call.func, ast.Attribute):
        return current_columns

    method_name = call.func.attr
    if method_name in {"reset_index", "query", "sort_values", "dropna", "copy", "astype", "set_index"}:
        return current_columns

    if method_name == "rename":
        mapping = None
        for keyword in call.keywords:
            if keyword.arg == "columns":
                mapping = _resolve_dict_expr(keyword.value, env, local_lists)
                break
        if mapping and current_columns is not None:
            return [mapping.get(column, column) for column in current_columns]
        return current_columns

    if method_name == "drop":
        columns = None
        for keyword in call.keywords:
            if keyword.arg == "columns":
                columns = _resolve_list_expr(keyword.value, env, local_lists, current_columns=current_columns)
                break
        if columns and current_columns is not None:
            return [column for column in current_columns if column not in columns]
        return current_columns

    return current_columns


def derive_method_columns(method: Any, module: Any) -> list[str] | None:
    try:
        source = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return None

    env = dict(vars(module))
    local_lists: dict[str, list[str]] = {}
    df_columns: dict[str, list[str] | None] = {}
    return_columns: list[str] | None = None

    def process_statements(statements: list[ast.stmt]) -> None:
        nonlocal return_columns
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                value = stmt.value
                if isinstance(value, (ast.List, ast.Tuple, ast.BinOp, ast.ListComp, ast.Name)):
                    resolved_list = _resolve_list_expr(value, env, local_lists)
                    if resolved_list is not None:
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                local_lists[target.id] = resolved_list
                        continue

                if isinstance(value, ast.Call):
                    if (
                        isinstance(value.func, ast.Attribute)
                        and isinstance(value.func.value, ast.Name)
                        and value.func.value.id == "pd"
                        and value.func.attr == "DataFrame"
                    ):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                df_columns[target.id] = None
                        continue

                    if isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name):
                        base_name = value.func.value.id
                        if base_name in df_columns:
                            updated = _apply_dataframe_call(value, df_columns.get(base_name), env, local_lists)
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):
                                    df_columns[target.id] = updated
                            continue

                    if isinstance(value.func, ast.Name) and value.args and isinstance(value.args[0], ast.Name):
                        base_name = value.args[0].id
                        if base_name in df_columns:
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):
                                    df_columns[target.id] = df_columns.get(base_name)
                            continue

                if isinstance(value, ast.Name) and value.id in df_columns:
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            df_columns[target.id] = df_columns.get(value.id)
                    continue

                if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id in df_columns:
                    selected = _resolve_list_expr(value.slice, env, local_lists, current_columns=df_columns.get(value.value.id))
                    if selected is not None:
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                df_columns[target.id] = selected
                        continue
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            df_columns[target.id] = df_columns.get(value.value.id)
                    continue

                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.attr == "columns":
                        column_owner = target.value.id
                        if column_owner in df_columns:
                            resolved_columns = _resolve_list_expr(value, env, local_lists, current_columns=df_columns.get(column_owner))
                            if resolved_columns is not None:
                                df_columns[column_owner] = resolved_columns

            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    base_name = call.func.value.id
                    if base_name in df_columns:
                        updated = _apply_dataframe_call(call, df_columns.get(base_name), env, local_lists)
                        if any(keyword.arg == "inplace" and _resolve_scalar_expr(keyword.value, env, local_lists) for keyword in call.keywords):
                            df_columns[base_name] = updated

            elif isinstance(stmt, ast.If):
                process_statements(stmt.body)
                process_statements(stmt.orelse)

            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name) and stmt.value.id in df_columns:
                    return_columns = df_columns.get(stmt.value.id)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            process_statements(node.body)
            break

    return return_columns


def derive_docstring_columns(method: Any) -> list[str] | None:
    doc = inspect.cleandoc(inspect.getdoc(method) or "")
    if not doc:
        return None

    patterns = [
        r"columns:\s*([A-Za-z0-9_,\s]+)",
        r"cột:\s*([A-Za-z0-9_,\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, doc, flags=re.IGNORECASE)
        if not match:
            continue
        parts = [part.strip(" `.") for part in match.group(1).split(",")]
        columns = [part for part in parts if part and " " not in part]
        if len(columns) >= 2:
            return columns
    return None
