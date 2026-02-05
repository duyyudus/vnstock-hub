from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
import math
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError


@dataclass(frozen=True)
class LLMProvider:
    name: str
    base_url: str
    api_key: str
    model: str


class TradeItem(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    side: str
    quantity: float
    execution_date: Optional[str] = None


class PositionItem(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    quantity: float


class ImagePositionItem(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    average_cost: float
    quantity: Optional[float] = None


class ExtractionResult(BaseModel):
    trades: List[TradeItem] = []
    positions: List[PositionItem] = []


class ImageExtractionResult(BaseModel):
    positions: List[ImagePositionItem] = []


def _build_chat_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _extract_json_payload(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start:end + 1])


def _coalesce_positions(result: ExtractionResult) -> List[PositionItem]:
    if result.positions:
        return result.positions

    totals: dict[str, float] = {}
    for trade in result.trades:
        side = trade.side.strip().lower()
        if side not in {"buy", "sell"}:
            continue
        ticker = trade.ticker.strip().upper()
        if not ticker:
            continue
        qty = float(trade.quantity)
        if side == "sell":
            qty *= -1
        totals[ticker] = totals.get(ticker, 0.0) + qty

    return [PositionItem(ticker=ticker, quantity=qty) for ticker, qty in totals.items()]


def _coerce_number(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_image_positions(items: List[ImagePositionItem]) -> List[ImagePositionItem]:
    normalized: dict[str, ImagePositionItem] = {}
    conflicted: set[str] = set()

    for item in items:
        ticker = item.ticker.strip().upper()
        if not ticker:
            continue

        average_cost = _coerce_number(item.average_cost)
        if average_cost is None or average_cost <= 0:
            continue
        quantity = _coerce_number(item.quantity)
        if quantity is not None and quantity <= 0:
            quantity = None

        average_cost = round(average_cost, 4)
        if quantity is not None:
            quantity = round(quantity, 4)

        existing = normalized.get(ticker)
        if existing:
            if existing.average_cost != average_cost:
                conflicted.add(ticker)
                continue
            if existing.quantity is not None and quantity is not None and existing.quantity != quantity:
                conflicted.add(ticker)
                continue
            if existing.quantity is None and quantity is not None:
                existing.quantity = quantity
            continue

        normalized[ticker] = ImagePositionItem(
            ticker=ticker,
            average_cost=average_cost,
            quantity=quantity,
        )

    for ticker in conflicted:
        normalized.pop(ticker, None)

    result = list(normalized.values())
    result.sort(key=lambda item: item.ticker)
    return result


def _build_image_user_prompt() -> str:
    return (
        "Extract portfolio positions from this broker screenshot.\n"
        "Return strict JSON only with:\n"
        "{\"positions\": [{\"ticker\": string, \"average_cost\": number, \"quantity\": number|null}]}\n"
        "Rules:\n"
        "- Only extract average cost per share and ticker.\n"
        "- Quantity is optional and ONLY if clearly shown (e.g., \"Tong KL\", \"So luong\", \"Quantity\").\n"
        "- Ignore market price, P&L, market value, totals, and any unrelated columns.\n"
        "- If average cost is missing, do not include that ticker.\n"
        "- If data is insufficient, return {\"positions\": []}.\n"
    )


async def extract_positions(
    rows: List[List[str]],
    providers: List[LLMProvider],
    timeout_seconds: int,
    caller: Optional[str] = None,
) -> List[PositionItem]:
    if not providers:
        raise ValueError("No LLM providers configured")

    logger = logging.getLogger("vnstock_hub.llm")
    caller_name = caller or "unknown"
    prompt_rows = json.dumps(rows, ensure_ascii=True)
    system_prompt = (
        "You are a data extraction engine. You must return strict JSON only."
    )
    user_prompt = (
        "Extract portfolio positions or trade history from the table rows below.\n"
        "Return JSON with either:\n"
        "1) positions: [{\"ticker\": string, \"quantity\": number}]\n"
        "or 2) trades: [{\"ticker\": string, \"side\": \"buy\"|\"sell\", \"quantity\": number, \"execution_date\": string}]\n"
        "You may include both positions and trades, but positions are preferred if present.\n"
        "If data is insufficient, return {\"positions\": []}.\n\n"
        f"Rows (array of arrays):\n{prompt_rows}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Optional[Exception] = None
    for provider in providers:
        try:
            if not provider.api_key:
                raise ValueError(f"Missing API key for provider {provider.name}")

            url = _build_chat_url(provider.base_url)
            logger.info(
                "llm_call provider=%s model=%s caller=%s",
                provider.name,
                provider.model,
                caller_name,
            )
            payload = {
                "model": provider.model,
                "messages": messages,
                "temperature": 0.2,
            }
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices") or []
            if not choices:
                raise ValueError("LLM response missing choices")
            content = (
                choices[0]
                .get("message", {})
                .get("content")
            )
            if not content:
                raise ValueError("LLM response missing message content")

            raw_payload = _extract_json_payload(content)
            extraction = ExtractionResult.model_validate(raw_payload)
            positions = _coalesce_positions(extraction)
            logger.info(
                "llm_success provider=%s positions=%s trades=%s caller=%s",
                provider.name,
                len(extraction.positions),
                len(extraction.trades),
                caller_name,
            )
            return positions
        except (httpx.HTTPError, ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "llm_failure provider=%s error=%s caller=%s",
                provider.name,
                exc,
                caller_name,
            )
            continue

    if last_error:
        raise last_error
    raise ValueError("Unable to extract positions from LLM providers")


async def extract_positions_from_image(
    image_bytes: bytes,
    mime_type: str,
    providers: List[LLMProvider],
    timeout_seconds: int,
    caller: Optional[str] = None,
) -> List[ImagePositionItem]:
    if not providers:
        raise ValueError("No LLM providers configured")

    logger = logging.getLogger("vnstock_hub.llm")
    caller_name = caller or "unknown"
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    system_prompt = "You are a data extraction engine. You must return strict JSON only."
    user_prompt = _build_image_user_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    last_error: Optional[Exception] = None
    for provider in providers:
        try:
            if not provider.api_key:
                raise ValueError(f"Missing API key for provider {provider.name}")

            url = _build_chat_url(provider.base_url)
            logger.info(
                "llm_call provider=%s model=%s caller=%s mode=image",
                provider.name,
                provider.model,
                caller_name,
            )
            payload = {
                "model": provider.model,
                "messages": messages,
                "temperature": 0.0,
            }
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices") or []
            if not choices:
                raise ValueError("LLM response missing choices")
            content = (
                choices[0]
                .get("message", {})
                .get("content")
            )
            if not content:
                raise ValueError("LLM response missing message content")

            raw_payload = _extract_json_payload(content)
            extraction = ImageExtractionResult.model_validate(raw_payload)
            positions = _normalize_image_positions(extraction.positions)
            logger.info(
                "llm_success provider=%s positions=%s caller=%s mode=image",
                provider.name,
                len(positions),
                caller_name,
            )
            return positions
        except (httpx.HTTPError, ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "llm_failure provider=%s error=%s caller=%s mode=image",
                provider.name,
                exc,
                caller_name,
            )
            continue

    if last_error:
        raise last_error
    raise ValueError("Unable to extract positions from LLM providers")
