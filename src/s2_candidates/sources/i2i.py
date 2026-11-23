from __future__ import annotations

import polars as pl

from ...config import DAY
from ...s1_features.past import BaseFeatures, PastEvents
from ...utils import log, rank_in_user
from ..config import I2I_DAYS, I2I_MIN_MEETINGS, I2I_SEEDS, I2I_TOPK, I2I_WINDOW, N_I2I


def _recent_sequences(listens: pl.DataFrame, T: int) -> pl.DataFrame:
    """Прослушивания за I2I_DAYS дней до T по порядку; несколько прослушиваний трека подряд считаются одним."""

    # только последение 30 дней
    seq = listens.filter(pl.col("timestamp") >= T - I2I_DAYS * DAY).select("uid", "item_id")
    new_item = pl.col("item_id").ne_missing(pl.col("item_id").shift(1))
    new_user = pl.col("uid").ne_missing(pl.col("uid").shift(1))

    # оставить строку/пару если сменился трек или пользователь
    return seq.filter(new_item | new_user)


def _pairs_at_distance(seq: pl.DataFrame, k: int) -> pl.DataFrame:
    """Пары (item, neighbour): трек и тот, что пользователь слушал через k треков после него; вес пары 1/k."""
    return (
        seq.select(
            pl.col("item_id").alias("item"),
            pl.col("item_id").shift(-k).alias("neighbour"),  # трек, идущий через k строк вниз
            (pl.col("uid") == pl.col("uid").shift(-k)).alias("same_user"),  # окно не должно выходить за пользователя
        )
        .filter(pl.col("same_user") & (pl.col("item") != pl.col("neighbour")))
        .select("item", "neighbour", pl.lit(1.0 / k, dtype=pl.Float32).alias("weight"))
    )


def _count_pairs(seq: pl.DataFrame) -> pl.DataFrame:
    """Пары треков, прослушанных не дальше I2I_WINDOW друг от друга: сумма весов 1/расстояние, от 2 встреч."""
    pairs = pl.concat([_pairs_at_distance(seq, k)
                       for k in range(1, I2I_WINDOW + 1)])

    # отзеркалим, потому что если после A слушали B, то до B случали А
    mirrored = pairs.select(pl.col("neighbour").alias("item"),
                            pl.col("item").alias("neighbour"),
                            "weight"
                            )
    pairs = (
        # соединяем все соседние прослушивания
        pl.concat([pairs, mirrored])
        .group_by(["item", "neighbour"])
        .agg(pl.col("weight").sum(), # складываем веса всех встреч пары A - B
             pl.len().alias("n_meetings") # сколько раз пара встретилась у разных пользователей
             )
    )
    return pairs.filter(pl.col("n_meetings") >= I2I_MIN_MEETINGS).drop("n_meetings")


def _normalize(pairs: pl.DataFrame, seq: pl.DataFrame) -> pl.DataFrame:
    """sim = weight / sqrt(слушали item * слушали neighbour), чтобы хиты не были соседями всех; топ-I2I_TOPK.

    :param seq: последовательность за последние 30 дней."""
    plays = (seq # uid | item_id
             .group_by("item_id")
             .agg(pl.len().cast(pl.Float32).alias("plays"))
             ) # item_id | plays
    item_plays = plays.rename({"item_id": "item", "plays": "item_plays"}) # item | item_plays
    neighbour_plays = plays.rename({"item_id": "neighbour", "plays": "neighbour_plays"}) # neighbour | neighbour_plays

    cosine = (pl.col("weight") / (pl.col("item_plays") * pl.col("neighbour_plays")).sqrt()).cast(pl.Float32)

    pairs = (pairs
             # для каждой пары для колонки item берет количество его прослушиваний и записывает в item_plays
             .join(item_plays, on="item")
             # для каждой пары для колонки neighbour берет количество его прослушиваний и записывает в neighbour_plays
             .join(neighbour_plays, on="neighbour"))

    # оставляем три колонки
    pairs = pairs.select("item", "neighbour", cosine.alias("sim"))
    # номер строки по убыванию sim: 1 у наибольшего
    rank_among_neighbours = (pl.col("sim")
                             .rank("ordinal", descending=True).over("item"))
    return (pairs
            .sort(["item", "neighbour"]) # расставляет строки по item, а внутри по neighbour
            .filter(rank_among_neighbours <= I2I_TOPK)) # берем топ 50 соседей по rank функции


def build_i2i(listens: pl.DataFrame, T: int) -> pl.DataFrame:
    """Таблица соседей (item, neighbour, sim): для каждого трека до I2I_TOPK треков, которые слушают рядом."""
    seq = _recent_sequences(listens, T)
    pairs = _count_pairs(seq)
    return _normalize(pairs, seq)


def i2i_scores(listens: pl.DataFrame, neighbours: pl.DataFrame) -> pl.DataFrame:
    """Оценки треков-соседей последних I2I_SEEDS треков пользователя; соседи более свежих треков весят больше.

    :param neighbours: Сходство для каждой пары треков: трек и его сосед.
    :return: Насколько трек подходит пользователю: uid, item_id, i2i_score (взвешенная сумма sim),
        i2i_max (наибольший sim), i2i_nseeds (соседом скольких последних треков пользователя трек оказался)
    """
    place_from_last = pl.col("timestamp").rank("ordinal", descending=True).over("uid")
    freshness = (-(pl.col("place").cast(pl.Float32) - 1) / 10.0).exp().cast(pl.Float32)  # 1, 0.90, 0.82, ...
    seeds = (
        listens.select("uid", "item_id", "timestamp")
        .group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max())
        .sort(["uid", "item_id"])
        .with_columns(place_from_last.alias("place"))
        .filter(pl.col("place") <= I2I_SEEDS) # последние 50 различных треков пользователя
        .with_columns(freshness.alias("seed_weight"))
        .select("uid", pl.col("item_id").alias("item"), "seed_weight")
    )
    return (
        seeds.join(neighbours, on="item") # item | neighbour | sim
        .group_by(["uid", "neighbour"]) # сворачиваем по соседу
        .agg(
            # вес ранга трека пользователя A умножаем на sim для пары A и X
            (pl.col("seed_weight") * pl.col("sim")).sum().alias("i2i_score"),
            pl.col("sim").max().alias("i2i_max"),
            pl.len().cast(pl.Int16).alias("i2i_nseeds"),
        )
        .rename({"neighbour": "item_id"})
        .sort(["uid", "item_id"])
    )


class I2ISource:
    """Источник i2i: треки, которые часто слушают рядом с последними треками пользователя, и признаки i2i_*."""

    name = "i2i"

    def __init__(self, past: PastEvents, base: BaseFeatures):
        """Строит таблицу соседей по всем пользователям и считает i2i-оценки для целевых."""
        neighbours = build_i2i(past.listens, past.T)
        log(f"  i2i: {neighbours.height:,} pairs")
        self.scores = i2i_scores(past.user_listens, neighbours)
        self.user_item = base.user_item

    def candidates(self) -> pl.DataFrame:
        """До N_I2I треков на пользователя с наибольшим i2i_score среди тех, которых он ещё не слушал."""
        return (
            self.scores.join(self.user_item.select("uid", "item_id"), on=["uid", "item_id"], how="anti")
            .filter(rank_in_user("i2i_score") <= N_I2I)
            .select("uid", "item_id")
        )

    def features(self, cands: pl.DataFrame) -> pl.DataFrame:
        """Добавляет i2i_score, i2i_max и i2i_nseeds (см. i2i_scores); трек не сосед последних треков - пусто."""
        return cands.join(self.scores, on=["uid", "item_id"], how="left", maintain_order="left")
