from __future__ import annotations

import polars as pl

from ...s1_features.past import BaseFeatures, PastEvents
from ..config import N_POP


class PopSource:
    """Источник pop: общий для всех список популярных за 7 дней треков; у пользователей без истории единственный."""

    name = "pop"

    def __init__(self, past: PastEvents, base: BaseFeatures):
        """Запоминает пользователей и N_POP самых популярных треков."""
        self.users = past.users
        self.items = pl.DataFrame({"item_id": pl.Series(base.pop_items[:N_POP], dtype=pl.Int32)})

    def candidates(self) -> pl.DataFrame:
        """Все пары «пользователь × популярный трек»."""
        return self.users.join(self.items, how="cross")

    def features(self, cands: pl.DataFrame) -> pl.DataFrame:
        """Ничего не добавляет: популярность трека уже есть в общих признаках i_*."""
        return cands
