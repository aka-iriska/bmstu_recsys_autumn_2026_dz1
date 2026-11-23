from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS

NON_FEATURES = {"uid", "item_id", "label"}


def feature_cols(df: pl.DataFrame) -> list[str]:
    """Названия признаков: все колонки таблицы кандидатов, кроме uid, item_id и label."""
    return [c for c in df.columns if c not in NON_FEATURES]


def only_users_with_pos(df: pl.DataFrame) -> pl.DataFrame:
    """Оставляет только пользователей, у которых среди кандидатов есть хотя бы один трек с label = 1."""
    pos_users = df.filter(pl.col("label") == 1).select("uid").unique()
    return df.join(pos_users, on="uid", how="semi").sort(["uid", "item_id"])


def to_lgb(frames: list[pl.DataFrame], feats: list[str], params: dict | None = None) -> lgb.Dataset:
    """lgb.Dataset для LambdaRank из таблиц кандидатов; пользователи без верных кандидатов отбрасываются.

    :param frames: таблицы кандидатов, например два временных среза; один пользователь в двух срезах - две группы
    :param params: параметры набора данных LightGBM, например {"feature_pre_filter": False} для подбора параметров
    """
    frames = [only_users_with_pos(f) for f in frames]
    groups = np.concatenate([f.group_by("uid", maintain_order=True).len()["len"].to_numpy() for f in frames])
    X = np.vstack([f.select(feats).to_numpy().astype(np.float32) for f in frames])
    label = np.concatenate([f["label"].to_numpy() for f in frames])
    return lgb.Dataset(
        data=X,
        label=label,
        group=groups,
        feature_name=feats,
        params=params,
        free_raw_data=True,
    )


def train_ranker(
    train: list[pl.DataFrame], early_stop: list[pl.DataFrame] | None = None, rounds: int = MAX_ROUNDS
) -> lgb.Booster:
    """Обучает LambdaRank на таблицах кандидатов train; пользователи без верных кандидатов не участвуют.

    :param train: таблицы кандидатов для обучения, например срезы A и B
    :param early_stop: таблицы кандидатов для ранней остановки; None - учить ровно rounds деревьев
    """

    feats = feature_cols(train[0])
    dtrain = to_lgb(train, feats)
    if early_stop is None:
        return lgb.train(LGB_PARAMS, dtrain, num_boost_round=rounds)
    dvalid = to_lgb(early_stop, feats)
    return lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.log_evaluation(50),
                   lgb.early_stopping(EARLY_STOPPING)],
    )


def predict_scores(model: lgb.Booster, cands: pl.DataFrame) -> pl.DataFrame:
    """Оценка ранкера для каждой строки кандидатов; считает по 2 млн строк, чтобы не занять всю память."""
    feats = model.feature_name()
    assert feats == feature_cols(cands), "признаки кандидатов не совпадают с признаками модели"
    scores = np.empty(cands.height, dtype=np.float32)
    step = 2_000_000
    for s in range(0, cands.height, step):
        X = cands.slice(s, step).select(feats).to_numpy().astype(np.float32)
        scores[s : s + step] = model.predict(X, num_threads=0)
    return cands.select("uid", "item_id").with_columns(pl.Series("score", scores))


def top_k_by(df: pl.DataFrame, col: str, k: int = 10) -> pl.DataFrame:
    """Каждому пользователю k треков с наибольшим col; rank - место, начиная с 0."""
    return (
        df.sort(["uid", "item_id"])
        .with_columns(pl.col(col).rank("ordinal", descending=True).over("uid").cast(pl.Int32).alias("rank"))
        .filter(pl.col("rank") <= k)
        .with_columns(pl.col("rank") - 1)
        .select("uid", "item_id", "rank")
    )


def fill_with_pop(recs: pl.DataFrame, users: pl.Series, pop_items: list, k: int = 10) -> pl.DataFrame:
    """Дополняет списки короче k популярными треками; пользователь без кандидатов получает только популярное."""
    users_df = pl.DataFrame({"uid": users.cast(pl.Int32)}).unique()
    pop_ids = pl.Series(pop_items[: 2 * k], dtype=pl.Int32)
    pop_ranks = np.arange(2 * k, dtype=np.int32) + 1000  # популярное всегда ниже собственных рекомендаций
    pop = pl.DataFrame({"item_id": pop_ids, "prank": pop_ranks})
    filler = users_df.join(pop, how="cross").join(recs, on=["uid", "item_id"], how="anti")
    own = recs.select("uid", "item_id", pl.col("rank").cast(pl.Int32).alias("r"))
    extra = filler.select("uid", "item_id", pl.col("prank").alias("r"))
    merged = pl.concat([own, extra]).join(users_df, on="uid", how="semi")
    merged = merged.with_columns((-pl.col("r")).alias("s"))
    return top_k_by(merged, "s", k)


def recommend(model: lgb.Booster, cands: pl.DataFrame, users: pl.Series, pop_items: list, k: int = 10) -> pl.DataFrame:
    """Итоговые k треков на пользователя: кандидаты по убыванию оценки ранкера, нехватка - популярными."""
    scores = predict_scores(model, cands)
    recs = top_k_by(scores, "score", k)
    return fill_with_pop(recs, users, pop_items, k)
