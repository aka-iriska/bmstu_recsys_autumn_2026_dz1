from __future__ import annotations

import argparse
import gc
import json

import polars as pl

from src.config import DAY, OUT_DIR, SEED
from src.s0_data.data import (
    earlier_boundary,
    load_test_users,
    load_train,
    make_truth,
    split_users,
    time_split,
    validation_boundary,
)
from src.s2_candidates.candidates import build_candidates
from src.s3_ranker.config import BEST_PARAMS_PATH, FINAL_ROUNDS_MULT, LGB_PARAMS, MAX_ROUNDS, TUNE_TRIALS
from src.s3_ranker.ranker import Ensemble, feature_cols, fill_with_pop, recommend, top_k_by, train_ensemble
from src.s3_ranker.tuning import tune_ranker
from src.s4_evaluation.metrics import ndcg_at_k, recall_of_candidates, source_report
from src.s4_evaluation.submission import submission_comment, write_submission
from src.utils import log

FOLD_A_PATH = OUT_DIR / "fold_a.parquet"  # кандидаты среза A (раньше): учат ранкер
FOLD_B_PATH = OUT_DIR / "fold_b.parquet"  # кандидаты среза B (позже): ранняя остановка и оценка
VAL_RESULT = OUT_DIR / "val_result.json"
TRUTH_PATH = OUT_DIR / "truth.parquet"  # правильные ответы среза B
POP_PATH = OUT_DIR / "pop_items.json"
BASELINE_COLS = ("ui_w7", "ui_cnt", "als_score")  # бейзлайны: сортировка кандидатов по одному признаку


def of_users(df: pl.DataFrame, uids: list) -> pl.DataFrame:
    """Оставляет строки только указанных пользователей."""
    return df.filter(pl.col("uid").is_in(uids))


def stage0_validation_boundaries() -> tuple[pl.DataFrame, int, int]:
    """Читает train.parquet и возвращает его вместе с границами срезов: T_A (раньше) и T_B (позже)."""
    df = load_train()
    T_b = validation_boundary(df)
    T_a = earlier_boundary(df, T_b)
    end = int(df["timestamp"].max()) + 1
    log(f"train: {df.height:,} events, {df['uid'].n_unique():,} users, {df['item_id'].n_unique():,} items, "
        f"span {(df['timestamp'].max() - df['timestamp'].min()) / DAY:.1f} days")
    log(f"slice A: T_A={T_a}, labels from {(T_b - T_a) / DAY:.1f} days | "
        f"slice B: T_B={T_b}, labels from {(end - T_b) / DAY:.1f} days")
    return df, T_a, T_b


def stage2_labeled_candidates(past: pl.DataFrame, T: int, truth: pl.DataFrame) -> tuple[pl.DataFrame, list]:
    """Строит кандидатов по past, ставит им метки 0/1 из truth и печатает, сколько ответов нашёл каждый источник."""
    cands, pop_items = build_candidates(past, T, truth["uid"].unique())
    cands = cands.join(truth, on=["uid", "item_id"], how="left").with_columns(pl.col("label").fill_null(0))
    log(f"candidate recall {recall_of_candidates(cands, truth):.4f}")
    log(f"sources:\n{source_report(cands, truth)}")
    return cands, pop_items


def stage4_evaluate(model: Ensemble, cands: pl.DataFrame, truth: pl.DataFrame, pop_items: list) -> dict:
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


def log_importance(model: Ensemble, top: int = 20) -> None:
    """Печатает признаки, которые сильнее всего улучшили деревья ранкера (суммарный gain)."""
    gains = zip(model.feature_name(), model.feature_importance("gain"))
    imp = sorted(gains, key=lambda x: -x[1])
    top_list = ", ".join(f"{f}={v:.0f}" for f, v in imp[:top])
    log(f"top features: {top_list}")


def log_params() -> None:
    """Печатает параметры LightGBM и откуда они: подобранные режимом tune или по умолчанию."""
    source = "tuned" if BEST_PARAMS_PATH.exists() else "default"
    log(f"LightGBM params ({source}): {LGB_PARAMS}")


def log_model_input(cands: pl.DataFrame, rows: int = 5) -> None:
    """Печатает вход ранкера: размер таблицы кандидатов и несколько случайных строк, признаки по строкам."""
    n = min(rows, cands.height)
    names = [f"row {i + 1}" for i in range(n)]
    sample = cands.sample(n, seed=SEED).cast(pl.Float64)
    table = sample.transpose(include_header=True, header_name="feature", column_names=names)
    log(f"model input: {cands.height:,} rows x {len(feature_cols(cands))} features; {n} random rows:")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(table)


def build_slice(df: pl.DataFrame, T: int, end: int | None) -> tuple[pl.DataFrame, pl.DataFrame, list]:
    """Кандидаты среза: признаки по событиям до T, метки по окну [T, end); возвращает кандидатов, truth и популярное."""
    past, future = time_split(df, T, end)
    truth = make_truth(future)
    log(f"T={T}: past {past.height:,} events | future {future.height:,} events, {truth.height:,} positive pairs")
    cands, pop_items = stage2_labeled_candidates(past, T, truth)
    return cands, truth, pop_items


def build_validation_folds() -> None:
    """Строит и сохраняет кандидатов срезов A и B, правильные ответы B и популярное для быстрых запусков."""
    df, T_a, T_b = stage0_validation_boundaries()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands_a, _, _ = build_slice(df, T_a, T_b)
    cands_a.write_parquet(FOLD_A_PATH)
    del cands_a
    gc.collect()
    cands_b, truth_b, pop_items = build_slice(df, T_b, None)
    log_model_input(cands_b)
    cands_b.write_parquet(FOLD_B_PATH)
    truth_b.write_parquet(TRUTH_PATH)
    POP_PATH.write_text(json.dumps(pop_items))
    log(f"saved {FOLD_A_PATH}, {FOLD_B_PATH}, {TRUTH_PATH}, {POP_PATH}")


def load_validation_folds() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list]:
    """Читает сохранённых кандидатов срезов A и B, правильные ответы B и популярное."""
    log(f"reuse saved folds {FOLD_A_PATH}, {FOLD_B_PATH}")
    pop_items = json.loads(POP_PATH.read_text())
    return pl.read_parquet(FOLD_A_PATH), pl.read_parquet(FOLD_B_PATH), pl.read_parquet(TRUTH_PATH), pop_items


def evaluate_ranker(rounds: int) -> None:
    """Учит ранкер на срезе A, выбирает число деревьев и оценивает на срезе B; пишет val_result.json для submit."""
    log_params()
    cands_a, cands_b, truth, pop_items = load_validation_folds()
    tr_u, es_u, ev_u = split_users(truth["uid"].unique())
    log(f"users: train {len(tr_u):,} | early-stop {len(es_u):,} | eval {len(ev_u):,}")
    model = train_ensemble([of_users(cands_a, tr_u)], [of_users(cands_b, es_u)], rounds)
    scores = stage4_evaluate(model, of_users(cands_b, ev_u), of_users(truth, ev_u), pop_items)
    log_importance(model)
    VAL_RESULT.write_text(json.dumps({"best_iter": model.best_iteration, **scores}))


def validate(rounds: int, ranker_only: bool) -> None:
    """Режим validate: учит ранкер на срезе A, печатает локальный NDCG@10 на срезе B и сохраняет кандидатов."""
    if not ranker_only:
        build_validation_folds()
    gc.collect()
    evaluate_ranker(rounds)


def tune(trials: int) -> None:
    """Режим tune: Optuna подбирает параметры LightGBM на сохранённых кандидатах, результат в best_params.json."""
    cands_a, cands_b, truth, _ = load_validation_folds()
    tr_u, es_u, _ = split_users(truth["uid"].unique())
    best = tune_ranker([of_users(cands_a, tr_u)], [of_users(cands_b, es_u)], trials)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    log(f"best params saved to {BEST_PARAMS_PATH}, validate and submit will use them: {best}")


def stage3_final_ranker(final_rounds: int) -> Ensemble:
    """Учит итоговый ранкер на срезах A и B всех пользователей; деревьев - best_iter × FINAL_ROUNDS_MULT."""
    best = json.loads(VAL_RESULT.read_text())["best_iter"]
    rounds = final_rounds or max(int(best * FINAL_ROUNDS_MULT), 50)
    log_params()
    log(f"final ranker: {rounds} rounds on slices A and B of all users")
    folds = [pl.read_parquet(FOLD_A_PATH), pl.read_parquet(FOLD_B_PATH)]
    model = train_ensemble(folds, None, rounds)
    model.save_model(OUT_DIR / "ranker.txt")
    return model


def submit(final_rounds: int) -> None:
    """Режим submit: учит итоговый ранкер, строит кандидатов по всему train и пишет submission.csv."""
    if not (FOLD_A_PATH.exists() and FOLD_B_PATH.exists()):
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
    """Разбирает аргументы командной строки и запускает validate, tune или submit."""
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["validate", "submit", "tune"])
    p.add_argument("--rounds", type=int, default=MAX_ROUNDS, help="максимум деревьев при validate")
    p.add_argument("--final-rounds", type=int, default=0, help="деревьев при submit (0 = best_iter * 1.3)")
    p.add_argument("--ranker-only", action="store_true", help="validate на сохранённых кандидатах, без их пересборки")
    p.add_argument("--trials", type=int, default=TUNE_TRIALS, help="сколько наборов параметров перебирает tune")
    args = p.parse_args()
    if args.mode == "tune":
        tune(args.trials)
    elif args.mode == "validate":
        validate(args.rounds, args.ranker_only)
    else:
        submit(args.final_rounds)


if __name__ == "__main__":
    main()
