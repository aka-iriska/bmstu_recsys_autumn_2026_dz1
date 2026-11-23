from __future__ import annotations

import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

from ...config import DAY, LISTEN, SEED
from ...s1_features.past import BaseFeatures, PastEvents
from ...utils import log, rank_in_user
from ..config import (
    ALS_ALPHA,
    ALS_FACTORS,
    ALS_FULL_PCT,
    ALS_ITERS,
    ALS_LIKE_BONUS,
    ALS_RECENT,
    ALS_REG,
    ALS_SKIP_W,
    ALS_TAU_DAYS,
    N_ALS,
)


def als_confidence(events: pl.DataFrame, reactions: pl.DataFrame, T: int) -> pl.DataFrame:
    """Вес c_ui каждой пары для iALS: больше, если слушал часто, недавно, до конца или лайкнул; формула в README.

    :param reactions: uid, item_id, ui_liked, ui_disliked - текущие лайки и дизлайки всех пользователей
    """
    age = (T - pl.col("timestamp")) / DAY
    full = pl.col("played_ratio_pct").fill_null(100) >= ALS_FULL_PCT
    freshness = (-age / ALS_TAU_DAYS).exp()
    completeness = pl.when(full).then(1.0).otherwise(ALS_SKIP_W)
    listen_weight = freshness * completeness
    r = (
        events.filter(pl.col("event_type") == LISTEN)
        .group_by(["uid", "item_id"])
        .agg(listen_weight.sum().alias("r"))
    )
    r_with_like = pl.col("r").fill_null(0.0) + pl.col("ui_liked").fill_null(0) * ALS_LIKE_BONUS
    keep = (pl.col("ui_disliked").fill_null(0) == 0) & (pl.col("r") > 0)
    conf = (1.0 + ALS_ALPHA * pl.col("r").log1p()).cast(pl.Float32)
    return (
        r.join(reactions, on=["uid", "item_id"], how="full", coalesce=True)
        .with_columns(r_with_like)
        .filter(keep)
        .select("uid", "item_id", conf.alias("conf"))
        .sort(["uid", "item_id"])
    )


def lookup(ids: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Для каждого значения x: его позиция в отсортированном массиве ids и есть ли оно там вообще."""
    if len(ids) == 0:
        return np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=bool)
    pos = np.clip(np.searchsorted(ids, x), 0, len(ids) - 1)
    return pos, ids[pos] == x


def rowwise_dot(a: np.ndarray, ia: np.ndarray, b: np.ndarray, ib: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Для каждого j скалярное произведение a[ia[j]]·b[ib[j]], NaN где mask[j] ложна; считает по 1 млн строк."""
    out = np.full(len(mask), np.nan, dtype=np.float32)
    step = 1_000_000
    for s in range(0, len(mask), step):
        idx = np.nonzero(mask[s : s + step])[0] + s
        out[idx] = np.einsum("ij,ij->i", a[ia[idx]], b[ib[idx]])
    return out


def normalize_rows(m: np.ndarray) -> np.ndarray:
    """Делит каждую строку матрицы на её длину, чтобы длина стала 1."""
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class ALS:
    """Модель iALS из implicit; строка i матрицы - пользователь u_ids[i], столбец j - трек i_ids[j]."""

    def __init__(self, conf: pl.DataFrame, banned: pl.DataFrame):
        """Собирает разреженную матрицу весов пользователь × трек и обучает на ней iALS.

        :param conf: uid, item_id, conf - результат als_confidence
        :param banned: uid, item_id - дизлайкнутые пары: в матрице их нет, но рекомендовать их нельзя
        """
        self.u_ids = conf["uid"].unique().sort().to_numpy()
        self.i_ids = conf["item_id"].unique().sort().to_numpy()
        rows = np.searchsorted(self.u_ids, conf["uid"].to_numpy())
        cols = np.searchsorted(self.i_ids, conf["item_id"].to_numpy())
        shape = (len(self.u_ids), len(self.i_ids))
        self.mat = sp.csr_matrix((conf["conf"].to_numpy(), (rows, cols)), shape=shape, dtype=np.float32)
        with threadpool_limits(1, "blas"):  # implicit распараллеливает сам, потоки BLAS ему мешают
            self.model = AlternatingLeastSquares(
                factors=ALS_FACTORS, regularization=ALS_REG, iterations=ALS_ITERS,
                random_state=SEED, calculate_training_loss=False,
            )
            self.model.fit(self.mat, show_progress=False)
        self.user_vecs = np.asarray(self.model.user_factors, dtype=np.float32)
        self.item_vecs = np.asarray(self.model.item_factors, dtype=np.float32)
        self.item_vecs_n = normalize_rows(self.item_vecs)
        self.seen = self._seen_matrix(banned)

    def _seen_matrix(self, banned: pl.DataFrame) -> sp.csr_matrix:
        """Пары, которые recommend не должен выдавать: всё прослушанное плюс дизлайки."""
        rows, r_ok = lookup(self.u_ids, banned["uid"].to_numpy())
        cols, c_ok = lookup(self.i_ids, banned["item_id"].to_numpy())
        ok = r_ok & c_ok
        extra = sp.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (rows[ok], cols[ok])), shape=self.mat.shape)
        return (self.mat + extra).tocsr()

    def recommend(self, users: np.ndarray, n: int) -> pl.DataFrame:
        """Каждому пользователю n треков с наибольшим x_u·y_i, которые он не слушал и не дизлайкал."""
        rows, ok = lookup(self.u_ids, users)
        rows, users = rows[ok], users[ok]
        ids, _ = self.model.recommend(rows, self.seen[rows], N=n, filter_already_liked_items=True)
        valid = ids >= 0
        uid_per_cell = np.repeat(users, ids.shape[1])
        return pl.DataFrame({
            "uid": uid_per_cell[valid.ravel()].astype(np.int32),
            "item_id": self.i_ids[ids[valid]].astype(np.int32),
        })

    def session_vectors(self, recent: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Средний вектор последних треков каждого пользователя длины 1 - то, что он слушает сейчас.

        :param recent: uid, item_id - последние уникальные треки пользователей
        :return: отсортированные uid и матрица их векторов
        """
        rows, ok = lookup(self.i_ids, recent["item_id"].to_numpy())
        known_uids = recent["uid"].to_numpy()[ok]
        users, inv = np.unique(known_uids, return_inverse=True)
        vecs = np.zeros((len(users), self.item_vecs.shape[1]), dtype=np.float32)
        np.add.at(vecs, inv, self.item_vecs_n[rows[ok]])
        return users, normalize_rows(vecs)

    def pair_features(self, cands: pl.DataFrame, recent: pl.DataFrame) -> pl.DataFrame:
        """Добавляет als_score = x_u·y_i, als_rank (место по als_score) и als_sim_recent (см. session_vectors)."""
        u, i = cands["uid"].to_numpy(), cands["item_id"].to_numpy()
        ur, u_ok = lookup(self.u_ids, u)
        ir, i_ok = lookup(self.i_ids, i)
        s_users, s_vecs = self.session_vectors(recent)
        sr, s_ok = lookup(s_users, u)
        score = rowwise_dot(self.user_vecs, ur, self.item_vecs, ir, u_ok & i_ok)
        sim = rowwise_dot(s_vecs, sr, self.item_vecs_n, ir, s_ok & i_ok)
        score_col = pl.Series("als_score", score).fill_nan(None)
        sim_col = pl.Series("als_sim_recent", sim).fill_nan(None)
        cands = cands.with_columns(score_col, sim_col)
        return cands.with_columns(rank_in_user("als_score").alias("als_rank"))


def last_distinct_items(listens: pl.DataFrame, n: int) -> pl.DataFrame:
    """Каждому пользователю n разных треков, которые он слушал последними."""
    return (
        listens.group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").max())
        .sort(["uid", "item_id"])
        .filter(pl.col("timestamp").rank("ordinal", descending=True).over("uid") <= n)
        .select("uid", "item_id")
    )


class AlsSource:
    """Источник als: треки, которых пользователь не слушал, с наибольшим x_u·y_i, и признаки als_*."""

    name = "als"

    def __init__(self, past: PastEvents, base: BaseFeatures):
        """Обучает iALS на событиях до T и запоминает последние треки целевых пользователей."""
        t0 = time.time()
        banned = base.reactions.filter(pl.col("ui_disliked") == 1).select("uid", "item_id")
        self.model = ALS(als_confidence(past.events, base.reactions, past.T), banned)
        self.users = past.users["uid"].to_numpy()
        self.recent = last_distinct_items(past.user_listens, ALS_RECENT)
        rows, cols = self.model.mat.shape
        log(f"  iALS: {rows:,} users x {cols:,} items, nnz {self.model.mat.nnz:,}, {time.time() - t0:.0f}s")

    def candidates(self) -> pl.DataFrame:
        """До N_ALS треков на пользователя с наибольшим x_u·y_i среди непрослушанных и не дизлайкнутых."""
        return self.model.recommend(self.users, N_ALS)

    def features(self, cands: pl.DataFrame) -> pl.DataFrame:
        """Добавляет кандидатам als_score, als_rank и als_sim_recent (см. ALS.pair_features)."""
        return self.model.pair_features(cands, self.recent)
