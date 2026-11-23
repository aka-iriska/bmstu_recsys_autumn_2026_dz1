from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..config import DATA_DIR, EVENT_TYPE, LISTEN, SEED
from .config import USER_SPLIT, VAL_FUTURE_SHARE


def load_train(path: Path = DATA_DIR / "train.parquet") -> pl.DataFrame:
    """Читает train.parquet с компактными типами."""
    df = pl.read_parquet(path).select(
        pl.col("uid").cast(pl.Int32),
        pl.col("timestamp").cast(pl.Int64),
        pl.col("item_id").cast(pl.Int32),
        pl.col("is_organic").cast(pl.Int8),
        pl.col("played_ratio_pct").cast(pl.Float32),
        pl.col("track_length_seconds").cast(pl.Float32),
        pl.col("event_type").cast(pl.Utf8).cast(EVENT_TYPE), # приводим к EVENT_TYPE
    )
    print(df.head())
    return df.sort(["uid", "timestamp", "event_type", "item_id"], maintain_order=True)


def load_test_users(path: Path = DATA_DIR / "test_users.csv") -> pl.Series:
    """Пользователи, для которых нужен сабмит (тестовые)."""
    return pl.read_csv(path)["user_id"].cast(pl.Int32)


def validation_boundary(df: pl.DataFrame) -> int:
    """Момент T_val, после которого остаётся доля VAL_FUTURE_SHARE прослушиваний train."""

    ts = df.filter(pl.col("event_type") == LISTEN)["timestamp"]
    return int(ts.quantile(1 - VAL_FUTURE_SHARE, interpolation="nearest"))


def time_split(df: pl.DataFrame, T: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Делит события на past (< T) и future (>= T).

    По past для каждой пары «пользователь × трек» строим признаки:
    сколько раз и как давно пользователь слушал трек, насколько трек популярен, его скоры i2i и iALS.
    По future целевую метку: послушал он в действительности этот трек или нет.
    """

    past = df.filter(pl.col("timestamp") < T)
    future = df.filter(pl.col("timestamp") >= T)

    return past, future


def make_truth(future: pl.DataFrame) -> pl.DataFrame:
    """Уникальные пары (uid, item_id), прослушанные в future, с label = 1."""
    return (
        future.filter(pl.col("event_type") == LISTEN)
        .select("uid", "item_id")
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    )


def split_users(uids: pl.Series) -> tuple[list, list, list]:
    """Случайно делит пользователей на обучение ранкера, раннюю остановку и оценку в пропорции USER_SPLIT."""
    all_users = np.random.default_rng(SEED).permutation(uids.unique().sort().to_numpy())
    reranker_users = int(USER_SPLIT[0] * len(all_users))
    es_users = int((USER_SPLIT[0] + USER_SPLIT[1]) * len(all_users))
    return (all_users[:reranker_users].tolist(),
            all_users[reranker_users:es_users].tolist(),
            all_users[es_users:].tolist())
