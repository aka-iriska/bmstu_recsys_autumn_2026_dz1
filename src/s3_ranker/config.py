from ..config import SEED

LGB_PARAMS = dict(
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
MAX_ROUNDS = 2000
EARLY_STOPPING = 100
FINAL_ROUNDS_MULT = 1.15  # финальный ранкер учится на всех пользователях среза -> чуть больше деревьев
