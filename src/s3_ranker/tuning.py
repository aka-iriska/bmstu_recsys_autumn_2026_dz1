from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import optuna
import polars as pl

from ..config import SEED
from ..utils import log
from .config import EARLY_STOPPING, LGB_PARAMS, MAX_ROUNDS, TUNE_SPACE
from .ranker import feature_cols, to_lgb


def _suggest(trial: optuna.Trial) -> dict:
    """Параметры одного испытания: каждый берётся из своего диапазона в TUNE_SPACE."""
    params = {}
    for name, (low, high, log_scale) in TUNE_SPACE.items():
        if isinstance(low, int):
            params[name] = trial.suggest_int(name, low, high, log=log_scale)
        else:
            params[name] = trial.suggest_float(name, low, high, log=log_scale)
    return params


def _objective(dtrain: lgb.Dataset, dvalid: lgb.Dataset) -> Callable[[optuna.Trial], float]:
    """Функция испытания: учит ранкер с параметрами trial и возвращает NDCG@10 на группе ранней остановки."""
    def objective(trial: optuna.Trial) -> float:
        params = {**LGB_PARAMS, **_suggest(trial)}
        stop = lgb.early_stopping(EARLY_STOPPING, verbose=False)
        model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid], callbacks=[stop])
        trial.set_user_attr("best_iter", model.best_iteration)
        return model.best_score["valid_0"]["ndcg@10"]
    return objective


def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Печатает результат испытания и лучший результат на данный момент."""
    log(f"trial {trial.number}: ndcg@10 {trial.value:.5f}, trees {trial.user_attrs['best_iter']}, "
        f"best so far {study.best_value:.5f}; {trial.params}")


def tune_ranker(train: list[pl.DataFrame], early_stop: list[pl.DataFrame], trials: int) -> dict:
    """Подбирает параметры LightGBM через Optuna по NDCG@10 на таблицах ранней остановки.

    :return: лучшие параметры и число деревьев best_iter, которое они дали
    """
    feats = feature_cols(train[0])
    dataset_params = {"feature_pre_filter": False}  # позволяет менять min_data_in_leaf без пересборки набора
    dtrain = to_lgb(train, feats, dataset_params)
    dvalid = to_lgb(early_stop, feats, dataset_params)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({name: LGB_PARAMS[name] for name in TUNE_SPACE})  # первым проверяются текущие параметры
    study.optimize(_objective(dtrain, dvalid), n_trials=trials, callbacks=[_log_trial])
    return {**study.best_params, "best_iter": study.best_trial.user_attrs["best_iter"]}
