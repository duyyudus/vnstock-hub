from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from app.core.config import settings


@dataclass(frozen=True)
class BrokerProfile:
    id: str
    name: str
    sheet: Optional[str]
    top_left: str
    bottom_right: str
    average_cost_multiplier: float


def _load_settings_yaml() -> dict:
    path = Path(settings.settings_yaml_path)
    if not path.exists():
        raise ValueError(f"settings.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("settings.yaml must be a YAML mapping")
    return data


def list_brokers() -> List[BrokerProfile]:
    data = _load_settings_yaml()
    raw_brokers = data.get("brokers")
    if raw_brokers is None:
        raise ValueError("settings.yaml missing 'brokers' section")
    if not isinstance(raw_brokers, list):
        raise ValueError("'brokers' must be a list")

    brokers: List[BrokerProfile] = []
    for item in raw_brokers:
        if not isinstance(item, dict):
            raise ValueError("Each broker entry must be a mapping")
        broker_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        top_left = str(item.get("top_left", "")).strip()
        bottom_right = str(item.get("bottom_right", "")).strip()
        sheet = item.get("sheet")
        sheet_value = str(sheet).strip() if sheet is not None else None
        multiplier_raw = item.get("average_cost_multiplier", 1)

        if not broker_id or not name or not top_left or not bottom_right:
            raise ValueError("Broker entries require id, name, top_left, bottom_right")
        try:
            average_cost_multiplier = float(multiplier_raw)
        except (TypeError, ValueError):
            raise ValueError("average_cost_multiplier must be a number") from None
        if average_cost_multiplier <= 0:
            raise ValueError("average_cost_multiplier must be greater than zero")

        brokers.append(BrokerProfile(
            id=broker_id,
            name=name,
            sheet=sheet_value,
            top_left=top_left,
            bottom_right=bottom_right,
            average_cost_multiplier=average_cost_multiplier,
        ))

    return brokers


def get_broker(broker_id: str) -> Optional[BrokerProfile]:
    broker_id = broker_id.strip().lower()
    for broker in list_brokers():
        if broker.id.lower() == broker_id:
            return broker
    return None
