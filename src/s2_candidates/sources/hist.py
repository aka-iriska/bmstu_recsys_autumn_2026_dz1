from __future__ import annotations

import polars as pl

from ...s1_features.past import BaseFeatures, PastEvents
from ..config import N_HIST


class HistSource:
    """Источник hist: треки, которые пользователь уже слушал больше всего и чаще всего недавно."""

    name = "hist"

    def __init__(self, past: PastEvents, base: BaseFeatures):
        """Запоминает признаки пар, по которым отбираются треки истории."""
        self.user_item = base.user_item

    def candidates(self) -> pl.DataFrame:
        """Трек берётся, если он в топ-N_HIST по ui_w7, в топ-N_HIST/2 по ui_w30 или в топ-N_HIST/3 по ui_cnt."""
        return self.user_item.filter(
            (pl.col("hist_rank_w7") <= N_HIST)
            | (pl.col("hist_rank_w30") <= N_HIST // 2)
            | (pl.col("hist_rank_cnt") <= N_HIST // 3)
        ).select("uid", "item_id")

    def features(self, cands: pl.DataFrame) -> pl.DataFrame:
        """Ничего не добавляет: признаки истории ui_* общие и уже есть в BaseFeatures."""
        return cands
