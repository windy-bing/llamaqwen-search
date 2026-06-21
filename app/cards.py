from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DailyCard:
    id: str
    title: str
    text: str
    source: str
    reference: str
    action: str


class CardDeck:
    def __init__(self, cards_file: Path) -> None:
        self.cards_file = cards_file
        self.cards: list[DailyCard] = []

    def load(self) -> None:
        if not self.cards_file.exists():
            raise FileNotFoundError(f"Cards file not found: {self.cards_file}")

        raw_cards = json.loads(self.cards_file.read_text(encoding="utf-8"))
        self.cards = [DailyCard(**item) for item in raw_cards]
        if not self.cards:
            raise ValueError("Cards file is empty.")

    def today(self, user_key: str = "default", today: date | None = None) -> DailyCard:
        if not self.cards:
            self.load()

        current_date = today or date.today()
        seed = stable_seed(f"{current_date.isoformat()}:{user_key}")
        return self.cards[seed % len(self.cards)]

    def draw(self, exclude_id: str | None = None) -> DailyCard:
        if not self.cards:
            self.load()
        candidates = [card for card in self.cards if card.id != exclude_id]
        return random.choice(candidates or self.cards)


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def card_to_dict(card: DailyCard) -> dict[str, Any]:
    return {
        "id": card.id,
        "title": card.title,
        "text": card.text,
        "source": card.source,
        "reference": card.reference,
        "action": card.action,
    }
