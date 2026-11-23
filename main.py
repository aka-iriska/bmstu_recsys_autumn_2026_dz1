from __future__ import annotations

import argparse
import gc
import json

import lightgbm as lgb
import polars as pl

from src.config import DAY, OUT_DIR, SEED
from src.s0_data.data import load_test_users, load_train, make_truth, split_users, time_split, validation_boundary
from src.s2_candidates.candidates import build_candidates
from src.s3_ranker.config import FINAL_ROUNDS_MULT, MAX_ROUNDS
from src.s3_ranker.ranker import feature_cols, fill_with_pop, recommend, top_k_by, train_ranker
from src.s4_evaluation.metrics import ndcg_at_k, recall_of_candidates, source_report
from src.s4_evaluation.submission import submission_comment, write_submission
from src.utils import log

FOLD_PATH = OUT_DIR / "train_fold.parquet"
VAL_RESULT = OUT_DIR / "val_result.json"
BASELINE_COLS = ("ui_w7", "ui_cnt", "als_score")  # бейзлайны: сортировка кандидатов по одному признаку


def of_users(df: pl.DataFrame, uids: list) -> pl.DataFrame:
    """Оставляет строки только указанных пользователей."""
    return df.filter(pl.col("uid").is_in(uids))


def stage0_validation_slice() -> tuple[pl.DataFrame, int, pl.DataFrame]:
    """Режет train.parquet по моменту T_val: события до него (past) и правильные ответы после него (truth)."""
    df = load_train()
    T = validation_boundary(df)
    past, future = time_split(df, T)
    truth = make_truth(future)
    log(f"train: {df.height:,} events, {df['uid'].n_unique():,} users, {df['item_id'].n_unique():,} items, "
        f"span {(df['timestamp'].max() - df['timestamp'].min()) / DAY:.1f} days")
    log(f"T_val={T}: past {past.height:,} events | future {future.height:,} events over "
        f"{(df['timestamp'].max() - T) / DAY:.1f} days, {truth.height:,} positive pairs")
    return past, T, truth


def stage2_labeled_candidates(past: pl.DataFrame, T: int, truth: pl.DataFrame) -> tuple[pl.DataFrame, list]:
    """Строит кандидатов по past, ставит им метки 0/1 из truth и печатает, сколько ответов нашёл каждый источник."""
    cands, pop_items = build_candidates(past, T, truth["uid"].unique())
    cands = cands.join(truth, on=["uid", "item_id"], how="left").with_columns(pl.col("label").fill_null(0))
    log(f"candidate recall {recall_of_candidates(cands, truth):.4f}")
    log(f"sources:\n{source_report(cands, truth)}")
    return cands, pop_items


def stage4_evaluate(model: lgb.Booster, cands: pl.DataFrame, truth: pl.DataFrame, pop_items: list) -> dict:
    """NDCG@10 ранкера и простых сортировок по одному признаку на пользователях, которых ранкер не видел."""
    users = truth["uid"].unique()
    ranker_recs = recommend(model, cands, users, pop_items)
    scores = {"ranker": ndcg_at_k(ranker_recs, truth)}
    for col in BASELINE_COLS:
        if col in cands.columns:
            scored = cands.filter(pl.col(col).is_not_null())
            recs = top_k_by(scored, col)
            recs = fill_with_pop(recs, users, pop_items)
            scores[f"only {col}"] = ndcg_at_k(recs, truth)
    log("EVAL NDCG@10: " + " | ".join(f"{k} {v:.4f}" for k, v in scores.items())
        + f" | best_iter {model.best_iteration}")
    return scores


def log_importance(model: lgb.Booster, top: int = 20) -> None:
    """Печатает признаки, которые сильнее всего улучшили деревья ранкера (суммарный gain)."""
    gains = zip(model.feature_name(), model.feature_importance("gain"))
    imp = sorted(gains, key=lambda x: -x[1])
    top_list = ", ".join(f"{f}={v:.0f}" for f, v in imp[:top])
    log(f"top features: {top_list}")


def log_model_input(cands: pl.DataFrame, rows: int = 5) -> None:
    """Печатает вход ранкера: размер таблицы кандидатов и несколько случайных строк, признаки по строкам."""
    n = min(rows, cands.height)
    names = [f"row {i + 1}" for i in range(n)]
    sample = cands.sample(n, seed=SEED).cast(pl.Float64)
    table = sample.transpose(include_header=True, header_name="feature", column_names=names)
    log(f"model input: {cands.height:,} rows x {len(feature_cols(cands))} features; {n} random rows:")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(table)


def validate(rounds: int) -> None:
    """Режим validate: учит ранкер на прошлом train, печатает локальный NDCG@10 и сохраняет кандидатов для submit."""
    past, T, truth = stage0_validation_slice()
    cands, pop_items = stage2_labeled_candidates(past, T, truth)
    log_model_input(cands)
    del past
    gc.collect()
    tr_u, es_u, ev_u = split_users(truth["uid"].unique())
    log(f"users: train {len(tr_u):,} | early-stop {len(es_u):,} | eval {len(ev_u):,}")
    train_cands = of_users(cands, tr_u)
    early_stop_cands = of_users(cands, es_u)
    model = train_ranker(train_cands, early_stop_cands, rounds)
    eval_cands = of_users(cands, ev_u)
    eval_truth = of_users(truth, ev_u)
    scores = stage4_evaluate(model, eval_cands, eval_truth, pop_items)
    log_importance(model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"best_iter": model.best_iteration, **scores}
    VAL_RESULT.write_text(json.dumps(result))
    cands.write_parquet(FOLD_PATH)
    log(f"saved {FOLD_PATH}")


def stage3_final_ranker(final_rounds: int) -> lgb.Booster:
    """Учит итоговый ранкер на кандидатах всех пользователей из validate; деревьев - best_iter × FINAL_ROUNDS_MULT."""
    best = json.loads(VAL_RESULT.read_text())["best_iter"]
    rounds = final_rounds or max(int(best * FINAL_ROUNDS_MULT), 50)
    log(f"final ranker: {rounds} rounds on the whole validation slice")
    fold = pl.read_parquet(FOLD_PATH)
    model = train_ranker(fold, None, rounds)
    model.save_model(str(OUT_DIR / "ranker.txt"))
    return model


def submit(final_rounds: int) -> None:
    """Режим submit: учит итоговый ранкер, строит кандидатов по всему train и пишет submission.csv."""
    if not FOLD_PATH.exists():
        raise SystemExit("Сначала запустите: uv run python main.py validate")
    model = stage3_final_ranker(final_rounds)
    df = load_train()
    T = int(df["timestamp"].max()) + 1
    test_users = load_test_users()
    known_users = df["uid"].unique().implode()
    n_cold = (~test_users.is_in(known_users)).sum()
    log(f"test users {test_users.len():,}, cold {n_cold}")
    cands, pop_items = build_candidates(df, T, test_users)
    log_model_input(cands)
    del df
    gc.collect()
    path = OUT_DIR / "submission.csv"
    recs = recommend(model, cands, test_users, pop_items)
    sub = write_submission(recs, test_users, path)
    log(f"saved {path} ({sub.height} rows)")
    comment = submission_comment(json.loads(VAL_RESULT.read_text()), model.num_trees(), cands)
    log(f"comment for Kaggle submission: {comment}")


def main() -> None:
    """Разбирает аргументы командной строки и запускает validate или submit."""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["validate", "submit"])
    p.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="максимум деревьев при validate")
    p.add_argument("--final-rounds", type=int, default=0, help="деревьев при submit (0 = best_iter * 1.15)")
    args = p.parse_args()
    if args.mode == "validate":
        validate(args.rounds)
    else:
        submit(args.final_rounds)


if __name__ == "__main__":
    main()
