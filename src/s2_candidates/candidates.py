from __future__ import annotations

import time

import polars as pl

from ..s1_features.past import BaseFeatures, compute_base_features, split_past
from ..utils import log
from .config import N_POP
from .sources.base import make_sources

# признаки пары, у которых «трека нет в истории» означает 0, а не пропуск
ZERO_FILL_PREFIXES = ("ui_cnt", "ui_w", "i2i_", "ui_liked", "ui_disliked", "ui_ndays")


def merge_sources(sources: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Склеивает пары всех источников без повторов; src_<имя> = 1, если пару предложил этот источник."""
    bits = {name: 1 << k for k, name in enumerate(sources)}
    tagged = [d.select("uid", "item_id", pl.lit(bits[n], dtype=pl.Int32).alias("src")) for n, d in sources.items()]
    flags = [((pl.col("src") & b) > 0).cast(pl.Int8).alias(f"src_{n}") for n, b in bits.items()]
    cands = (
        pl.concat(tagged)
        .group_by(["uid", "item_id"])
        .agg(pl.col("src").bitwise_or())
        .with_columns(flags)
        .drop("src")
        .sort(["uid", "item_id"])
    )
    n_users = max(cands["uid"].n_unique(), 1)
    sizes = ", ".join(f"{n}={d.height:,}" for n, d in sources.items())
    log(f"  candidates: {cands.height:,} ({cands.height / n_users:.0f}/user); {sizes}")
    return cands


def _share(num: str, den: str) -> pl.Expr:
    """num / den; знаменатель не меньше 1, чтобы не делить на 0."""
    return (pl.col(num) / pl.col(den).clip(lower_bound=1)).cast(pl.Float32)


def add_derived_features(cands: pl.DataFrame, pop_items: list) -> pl.DataFrame:
    """Добавляет in_hist (слушал ли трек), is_pop (в топе популярного) и ui_share_* (доля трека у пользователя)."""
    return cands.with_columns(
        pl.col("ui_cnt").is_not_null().cast(pl.Int8).alias("in_hist"),
        pl.col("item_id").is_in(pop_items[:N_POP]).cast(pl.Int8).alias("is_pop"),
        _share("ui_cnt_7d", "u_cnt_7d").alias("ui_share_7d"),
        _share("ui_cnt_30d", "u_cnt_30d").alias("ui_share_30d"),
        _share("ui_cnt", "u_cnt").alias("ui_share"),
        (pl.col("ui_last_age") - pl.col("u_last_age")).cast(pl.Float32).alias("ui_last_age_rel"),
    )


def fill_missing(cands: pl.DataFrame) -> pl.DataFrame:
    """Ставит 0 в счётчики, где пустое значение значит «ни разу», и сортирует по uid, item_id."""
    zero_cols = [c for c in cands.columns if c.startswith(ZERO_FILL_PREFIXES)] + ["i_likes", "i_dislikes"]
    return cands.with_columns([pl.col(c).fill_null(0) for c in zero_cols]).sort(["uid", "item_id"])


def attach_base_features(cands: pl.DataFrame, base: BaseFeatures) -> pl.DataFrame:
    """Добавляет кандидатам общие признаки пары, пользователя, трека и признаки, вычисленные из них."""
    for table in (base.user_item, base.reactions):
        cands = cands.join(table, on=["uid", "item_id"], how="left", maintain_order="left")
    cands = (
        cands.join(base.user, on="uid", how="left", maintain_order="left")
        .join(base.items, on="item_id", how="left", maintain_order="left")
        .join(base.item_reactions, on="item_id", how="left", maintain_order="left")
    )
    cands = add_derived_features(cands, base.pop_items)
    return fill_missing(cands)


def build_candidates(events: pl.DataFrame, T: int, users: pl.Series) -> tuple[pl.DataFrame, list]:
    """Каждому пользователю несколько сотен треков-кандидатов из всех источников со всеми признаками.

    :param events: все события до момента T
    :param users: пользователи, для которых нужны рекомендации
    :return: таблица кандидатов (uid, item_id, src_*, признаки) и список популярных треков
    """
    t0 = time.time()
    past = split_past(events, T, users)
    log("  features: item, user-item, user, reactions")
    base = compute_base_features(past)
    sources = make_sources(past, base)
    cands = merge_sources({s.name: s.candidates() for s in sources})
    for source in sources:
        cands = source.features(cands)
    cands = attach_base_features(cands, base)
    log(f"  candidates and features ready in {time.time() - t0:.0f}s")
    return cands, base.pop_items
