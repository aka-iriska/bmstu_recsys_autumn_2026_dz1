from __future__ import annotations

import polars as pl

from ...s1_features.config import ITEM_WINDOWS
from ...s1_features.past import BaseFeatures, PastEvents, popular_items
from ..config import N_POP


class PopSource:
    """Источник pop: общий список треков, популярных за 1, 7 и 30 дней; у пользователей без истории единственный."""

    name = "pop"

    def __init__(self, past: PastEvents, base: BaseFeatures):
        """Запоминает пользователей и объединение N_POP самых популярных треков по каждому окну."""
        self.users = past.users
        tops = [popular_items(base.items, d, N_POP) for d in ITEM_WINDOWS]
        item_ids = sorted(set().union(*tops))
        self.items = pl.DataFrame({"item_id": pl.Series(item_ids, dtype=pl.Int32)})

    def candidates(self) -> pl.DataFrame:
        """Все пары «пользователь × популярный трек»."""
        return self.users.join(self.items, how="cross")

    def features(self, cands: pl.DataFrame) -> pl.DataFrame:
        """Ничего не добавляет: популярность трека по окнам уже есть в общих признаках i_*."""
        return cands
