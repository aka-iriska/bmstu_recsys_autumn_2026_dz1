from __future__ import annotations

from typing import Protocol

import polars as pl

from ...s1_features.past import BaseFeatures, PastEvents
from ..config import SOURCES
from .als import AlsSource
from .hist import HistSource
from .i2i import I2ISource
from .pop import PopSource


class Source(Protocol):
    """Что умеет источник кандидатов: предложить пары (uid, item_id) и добавить кандидатам свои признаки."""

    name: str

    def candidates(self) -> pl.DataFrame:
        """Пары (uid, item_id), которые предлагает источник."""

    def features(self, cands: pl.DataFrame) -> pl.DataFrame:
        """Таблица кандидатов с добавленными колонками-признаками источника."""


SOURCE_CLASSES = {"hist": HistSource, "i2i": I2ISource, "als": AlsSource, "pop": PopSource}


def make_sources(past: PastEvents, base: BaseFeatures) -> list[Source]:
    """Создаёт источники из SOURCES в том же порядке."""
    return [SOURCE_CLASSES[name](past, base) for name in SOURCES]
