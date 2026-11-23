from __future__ import annotations

from pathlib import Path

import polars as pl

from ..s2_candidates.config import ALS_ALPHA, ALS_FACTORS, ALS_REG, I2I_DAYS, I2I_WINDOW, SOURCES


def write_submission(recs: pl.DataFrame, test_users: pl.Series, path: Path) -> pl.DataFrame:
    """Пишет submission.csv: user_id и до 10 item_id через пробел, от лучшего к худшему."""
    sub = (
        recs.sort(["uid", "rank"])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_id").cast(pl.Utf8).str.join(" ").alias("item_ids"))
        .rename({"uid": "user_id"})
    )
    sub = pl.DataFrame({"user_id": test_users}).join(sub, on="user_id", how="left")
    assert sub["item_ids"].null_count() == 0, "есть пользователи без рекомендаций"
    assert sub.height == test_users.len()
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_csv(path)
    return sub


def submission_comment(val: dict, n_trees: int, cands: pl.DataFrame) -> str:
    """Строка для поля Description при загрузке на Kaggle: локальная оценка и главные настройки.

    :param val: содержимое val_result.json из последнего validate
    :param n_trees: число деревьев итогового ранкера
    :param cands: кандидаты для test_users, по ним считается число кандидатов на пользователя
    """
    per_user = cands.height / max(cands["uid"].n_unique(), 1)
    parts = [
        f"local val NDCG@10 {val['ranker']:.4f}",
        f"trees {n_trees}",
        f"sources {'+'.join(SOURCES)}",
        f"{per_user:.0f} cand/user",
    ]
    if "als" in SOURCES:
        parts.append(f"iALS f{ALS_FACTORS} reg{ALS_REG:g} a{ALS_ALPHA:g}")
    if "i2i" in SOURCES:
        parts.append(f"i2i {I2I_DAYS}d win{I2I_WINDOW}")
    return " | ".join(parts)
