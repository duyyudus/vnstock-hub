from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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


class ExtractionResult(BaseModel):
    trades: List[TradeItem] = []
    positions: List[PositionItem] = []


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


async def extract_positions(
    rows: List[List[str]],
    providers: List[LLMProvider],
    timeout_seconds: int,
) -> List[PositionItem]:
    if not providers:
        raise ValueError("No LLM providers configured")

    logger = logging.getLogger("vnstock_hub.portfolio_import")
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
            logger.info("portfolio_import_llm_call provider=%s model=%s", provider.name, provider.model)
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
                "portfolio_import_llm_success provider=%s positions=%s trades=%s",
                provider.name,
                len(extraction.positions),
                len(extraction.trades),
            )
            return positions
        except (httpx.HTTPError, ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning("portfolio_import_llm_failure provider=%s error=%s", provider.name, exc)
            continue

    if last_error:
        raise last_error
    raise ValueError("Unable to extract positions from LLM providers")
