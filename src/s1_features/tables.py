from __future__ import annotations

import polars as pl

from ..config import DAY, DISLIKE, LIKE, LISTEN, UNDISLIKE, UNLIKE
from .config import UI_DECAYS, UI_WINDOWS


def age_days(T: int) -> pl.Expr:
    """Выражение: сколько дней прошло от события до момента T."""
    return ((T - pl.col("timestamp")) / DAY).cast(pl.Float32)


def _activity_aggs() -> list[pl.Expr]:
    """Выражения по паре: сколько раз и как давно пользователь слушал трек."""
    # Каждое выражение считается по всем прослушиваниям одной пары (uid, item_id), отсортированным по времени.
    # age - сколько дней назад было прослушивание, day - номер дня.
    return [
        # сколько раз пользователь слушал трек за всё время
        pl.len().cast(pl.Int32).alias("ui_cnt"),
        # сколько раз слушал за последние d дней: ui_cnt_1d, ui_cnt_3d, ui_cnt_7d, ...
        *[(pl.col("age") < d).sum().cast(pl.Int32).alias(f"ui_cnt_{d}d") for d in UI_WINDOWS],
        # счётчик, где каждое прослушивание весит exp(-давность / tau): свежие весят больше (ui_w1, ui_w7, ui_w30)
        *[(-pl.col("age") / tau).exp().sum().cast(pl.Float32).alias(f"ui_w{tau}") for tau in UI_DECAYS],
        # сколько дней назад было последнее прослушивание трека
        pl.col("age").min().alias("ui_last_age"),
        # сколько дней назад было первое прослушивание трека
        pl.col("age").max().alias("ui_first_age"),
        # сколько прослушиваний других треков было после последнего прослушивания этого (0 - он был последним)
        pl.col("pos_from_end").min().alias("ui_pos_from_end"),
        # в скольких разных днях слушал трек
        pl.col("day").n_unique().cast(pl.Int32).alias("ui_ndays"),
    ]


def _quality_aggs() -> list[pl.Expr]:
    """Выражения по паре: как слушал трек, то есть сколько доигрывал и доля прослушиваний с is_organic = 1."""
    pr = pl.col("played_ratio_pct")  # сколько процентов трека доиграно за прослушивание
    return [
        # сколько процентов трека в среднем доигрывал
        pr.mean().cast(pl.Float32).alias("ui_pr_mean"),
        # сколько процентов доиграл в последнее прослушивание
        pr.last().alias("ui_pr_last"),
        # доля прослушиваний, где доиграл не меньше 80% трека
        (pr >= 80).mean().cast(pl.Float32).alias("ui_full_share"),
        # доля прослушиваний, которые прервал раньше 30% трека (скипы)
        (pr < 30).mean().cast(pl.Float32).alias("ui_skip_share"),
        # доля органических прослушиваний (is_organic = 1)
        pl.col("is_organic").cast(pl.Float32).mean().alias("ui_organic"),
        # было ли последнее прослушивание органическим
        pl.col("is_organic").last().alias("ui_organic_last"),
    ]


def _user_item_aggs() -> list[pl.Expr]:
    """Все выражения по паре: активность (сколько и когда слушал) и качество (как слушал)."""
    return [*_activity_aggs(), *_quality_aggs()]


def user_item_features(listens: pl.DataFrame, T: int) -> pl.DataFrame:
    """Признаки пары (uid, item_id); прослушивания должны быть отсортированы по uid и времени."""
    lst = listens.with_columns(
        age_days(T).alias("age"),
        (pl.col("timestamp") // DAY).cast(pl.Int32).alias("day"),
        pl.int_range(pl.len(), dtype=pl.Int32).reverse().over("uid").alias("pos_from_end"),
    )
    aggs = _user_item_aggs()
    ui = lst.group_by(["uid", "item_id"]).agg(aggs).sort(["uid", "item_id"])
    listened_span = pl.col("ui_first_age") - pl.col("ui_last_age")
    # у трека с одним прослушиванием интервалов нет: null, а не 0
    mean_gap = pl.when(pl.col("ui_cnt") > 1).then(listened_span / (pl.col("ui_cnt") - 1)).cast(pl.Float32)
    age_over_gap = (pl.col("ui_last_age") / (pl.col("ui_mean_gap") + 0.01)).cast(pl.Float32)
    ui = ui.with_columns(mean_gap.alias("ui_mean_gap"))
    return ui.with_columns(age_over_gap.alias("ui_age_over_gap"))


def _last_state(react: pl.DataFrame, on: str, off: str, name: str) -> pl.DataFrame:
    """Для пары 1, если последнее из событий on/off - on: like после unlike → 1, unlike после like → 0."""
    return (
        react.filter(pl.col("event_type").is_in([on, off]))
        .group_by(["uid", "item_id"])
        .agg((pl.col("event_type").last() == on).cast(pl.Int8).alias(name))
    )


def reaction_features(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Лайкнут ли и дизлайкнут ли трек у пользователя сейчас и сколько всего лайков и дизлайков у трека.

    :return: (uid, item_id, ui_liked, ui_disliked), (item_id, i_likes, i_dislikes)
    """
    react = events.filter(pl.col("event_type") != LISTEN)
    like_state = _last_state(react, on=LIKE, off=UNLIKE, name="ui_liked")
    dislike_state = _last_state(react, on=DISLIKE, off=UNDISLIKE, name="ui_disliked")
    ui_react = like_state.join(dislike_state, on=["uid", "item_id"], how="full", coalesce=True)
    item_react = react.group_by("item_id").agg(
        (pl.col("event_type") == LIKE).sum().cast(pl.Int32).alias("i_likes"),
        (pl.col("event_type") == DISLIKE).sum().cast(pl.Int32).alias("i_dislikes"),
    )
    return ui_react, item_react


def user_features(listens: pl.DataFrame, T: int) -> pl.DataFrame:
    """Признаки пользователя: сколько слушает, как давно, сколько разных треков и как часто повторяет."""
    lst = listens.with_columns(age_days(T).alias("age"), (pl.col("timestamp") // DAY).alias("day"))
    u = lst.group_by("uid").agg(
        pl.len().cast(pl.Int32).alias("u_cnt"),
        pl.col("item_id").n_unique().cast(pl.Int32).alias("u_nitems"),
        *[(pl.col("age") < d).sum().cast(pl.Int32).alias(f"u_cnt_{d}d") for d in (1, 7, 30)],
        pl.col("age").min().alias("u_last_age"),
        pl.col("age").max().alias("u_first_age"),
        pl.col("day").n_unique().cast(pl.Int32).alias("u_ndays"),
        pl.col("played_ratio_pct").mean().cast(pl.Float32).alias("u_pr_mean"),
        pl.col("is_organic").cast(pl.Float32).mean().alias("u_organic"),
        pl.col("item_id").filter(pl.col("age") < 30).n_unique().cast(pl.Int32).alias("u_nitems_30d"),
    )
    return u.with_columns(
        (pl.col("u_cnt") / pl.col("u_nitems")).cast(pl.Float32).alias("u_repeat"),
        (pl.col("u_cnt_30d") / pl.col("u_nitems_30d").clip(lower_bound=1)).cast(pl.Float32).alias("u_repeat_30d"),
        (pl.col("u_cnt") / pl.col("u_ndays")).cast(pl.Float32).alias("u_per_day"),
    )


def item_features(listens: pl.DataFrame, T: int) -> pl.DataFrame:
    """Признаки трека по всем пользователям: сколько слушают по окнам, растёт ли, дослушивают ли, давно ли вышел."""
    lst = listens.with_columns(age_days(T).alias("age"))
    it = lst.group_by("item_id").agg(
        pl.len().cast(pl.Int32).alias("i_cnt"),
        *[(pl.col("age") < d).sum().cast(pl.Int32).alias(f"i_cnt_{d}d") for d in (1, 7, 30)],
        pl.col("uid").n_unique().cast(pl.Int32).alias("i_users"),
        pl.col("uid").filter(pl.col("age") < 7).n_unique().cast(pl.Int32).alias("i_users_7d"),
        pl.col("uid").filter(pl.col("age") < 30).n_unique().cast(pl.Int32).alias("i_users_30d"),
        pl.col("played_ratio_pct").mean().cast(pl.Float32).alias("i_pr_mean"),
        pl.col("is_organic").cast(pl.Float32).mean().alias("i_organic"),
        pl.col("age").min().alias("i_last_age"),
        pl.col("age").max().alias("i_first_age"),
        pl.col("track_length_seconds").max().alias("i_len"),
    )
    return it.sort("item_id").with_columns(
        (pl.col("i_cnt") / pl.col("i_users")).cast(pl.Float32).alias("i_repeat"),
        (pl.col("i_cnt_7d") / (pl.col("i_cnt_30d") + 1)).cast(pl.Float32).alias("i_trend_7_30"),
        pl.col("i_users_7d").rank("ordinal", descending=True).cast(pl.Int32).alias("i_pop_rank_7d"),
    )
