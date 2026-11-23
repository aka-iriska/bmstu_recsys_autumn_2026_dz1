import json
import os

from ..config import OUT_DIR, SEED

BEST_PARAMS_PATH = OUT_DIR / "best_params.json"  # результат режима tune; если файл есть, он перекрывает LGB_DEFAULTS
LGB_DEFAULTS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[10],
    lambdarank_truncation_level=20,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=SEED,
    num_threads=0,
)
ENSEMBLE_SIZE = int(os.environ.get("RECSYS_SEEDS", 1))  # сколько ранкеров с разными seed усредняется; 1 - один ранкер
MAX_ROUNDS = 2000
EARLY_STOPPING = 100
FINAL_ROUNDS_MULT = 1.3  # финальный ранкер учится на двух срезах всех пользователей, данных втрое больше
TUNE_TRIALS = 30  # сколько наборов параметров перебирает Optuna в режиме tune
TUNE_SPACE = {  # диапазоны поиска Optuna: (от, до, логарифмическая шкала)
    "num_leaves": (15, 255, True),
    "min_data_in_leaf": (20, 500, True),
    "learning_rate": (0.03, 0.1, True),
    "feature_fraction": (0.5, 1.0, False),
    "bagging_fraction": (0.5, 1.0, False),
    "lambda_l2": (0.001, 30.0, True),
    "lambdarank_truncation_level": (10, 50, False),
}


def _with_tuned(defaults: dict) -> dict:
    """Параметры по умолчанию, перекрытые подобранными режимом tune (best_params.json), если файл есть."""
    if not BEST_PARAMS_PATH.exists():
        return defaults
    tuned = json.loads(BEST_PARAMS_PATH.read_text())
    return {**defaults, **{name: tuned[name] for name in TUNE_SPACE if name in tuned}}


LGB_PARAMS = _with_tuned(LGB_DEFAULTS)
