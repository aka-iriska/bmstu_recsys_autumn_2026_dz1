from __future__ import annotations

import numpy as np
import polars as pl


def ndcg_at_k(recs: pl.DataFrame, truth: pl.DataFrame, k: int = 10) -> float:
    """NDCG@k, усреднённый по всем пользователям из truth; у кого нет ни одного попадания - 0.

    :param recs: uid, item_id, rank (0..k-1)
    :param truth: uid, item_id - что пользователь реально слушал
    """
    n_relevant = np.arange(1, k + 1, dtype=np.int64)
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    idcg = pl.DataFrame({"n": n_relevant, "idcg": np.cumsum(discounts)})
    gain = 1.0 / (pl.col("rank").cast(pl.Float64) + 2).log(2)
    dcg = (
        recs.filter(pl.col("rank") < k)
        .join(truth.select("uid", "item_id"), on=["uid", "item_id"], how="inner")
        .group_by("uid")
        .agg(gain.sum().alias("dcg"))
    )
    res = (
        truth.group_by("uid")
        .agg(pl.len().clip(upper_bound=k).cast(pl.Int64).alias("n"))
        .join(idcg, on="n", how="left")
        .join(dcg, on="uid", how="left")
        .with_columns(pl.col("dcg").fill_null(0.0))
    )
    return float((res["dcg"] / res["idcg"]).mean())


def recall_of_candidates(cands: pl.DataFrame, truth: pl.DataFrame) -> float:
    """Доля правильных пар (uid, item_id), попавших в кандидаты; выше этого ранкер ничего не найдёт."""
    hit = truth.join(cands.select("uid", "item_id"), on=["uid", "item_id"], how="semi").height
    return hit / max(truth.height, 1)


def source_report(cands: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """По источникам: сколько пар дал, долю найденных ответов, долю ответов среди своих, сколько нашёл только он."""
    src_cols = [c for c in cands.columns if c.startswith("src_")]
    n_src = pl.sum_horizontal([pl.col(c) for c in src_cols])
    rows = []
    for c in src_cols:
        sub = cands.filter(pl.col(c) == 1)
        only_this = cands.filter((pl.col(c) == 1) & (n_src == 1))
        rows.append({
            "source": c.removeprefix("src_"),
            "rows": sub.height,
            "recall": recall_of_candidates(sub, truth),
            "precision": float(sub["label"].mean() or 0.0),
            "unique_hits": int(only_this["label"].sum()),
        })
    return pl.DataFrame(rows)
