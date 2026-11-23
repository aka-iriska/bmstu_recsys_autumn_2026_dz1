from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import LISTEN
from ..utils import rank_in_user
from .config import POP_LIST_SIZE, POP_WINDOW
from .tables import item_features, reaction_features, user_features, user_item_features


@dataclass
class PastEvents:
    """События до момента T: все, только прослушивания и прослушивания пользователей, которым нужны рекомендации."""

    T: int
    events: pl.DataFrame  # все события всех пользователей
    listens: pl.DataFrame  # прослушивания всех пользователей
    user_listens: pl.DataFrame  # прослушивания целевых пользователей
    users: pl.DataFrame  # uid целевых пользователей без повторов


@dataclass
class BaseFeatures:
    """Признаки, которые считаются всегда, какие бы источники ни были включены."""

    items: pl.DataFrame  # i_*: треки по всем пользователям
    user_item: pl.DataFrame  # ui_* и hist_rank_*: пары целевых пользователей
    user: pl.DataFrame  # u_*: целевые пользователи
    reactions: pl.DataFrame  # ui_liked, ui_disliked: пары всех пользователей
    item_reactions: pl.DataFrame  # i_likes, i_dislikes
    pop_items: list  # item_id по убыванию популярности за POP_WINDOW дней: для добивки списков и признака is_pop


def split_past(events: pl.DataFrame, T: int, users: pl.Series) -> PastEvents:
    """Собирает PastEvents: отбирает прослушивания всех и прослушивания пользователей из users."""
    listens = events.filter(pl.col("event_type") == LISTEN)
    users_df = pl.DataFrame({"uid": users.cast(pl.Int32)}).unique()
    user_listens = listens.join(users_df, on="uid", how="semi", maintain_order="left")
    return PastEvents(T=T, events=events, listens=listens, user_listens=user_listens, users=users_df)


def popular_items(items: pl.DataFrame, window: int, size: int) -> list:
    """size самых популярных треков за window дней: по числу разных слушателей, при равенстве по прослушиваниям."""
    order = items.sort([f"i_users_{window}d", f"i_cnt_{window}d", "item_id"], descending=[True, True, False])
    return order.head(size)["item_id"].to_list()


def add_history_ranks(user_item: pl.DataFrame) -> pl.DataFrame:
    """Добавляет hist_rank_*: место трека среди треков пользователя по ui_w7, ui_w30, ui_w1 и ui_cnt (1 - первое)."""
    return user_item.with_columns(
        rank_in_user("ui_w7").alias("hist_rank_w7"),
        rank_in_user("ui_w30").alias("hist_rank_w30"),
        rank_in_user("ui_w1").alias("hist_rank_w1"),
        rank_in_user("ui_cnt").alias("hist_rank_cnt"),
    )


def compute_base_features(past: PastEvents) -> BaseFeatures:
    """Считает BaseFeatures по событиям до T."""
    items = item_features(past.listens, past.T)
    reactions, item_reactions = reaction_features(past.events)
    return BaseFeatures(
        items=items,
        user_item=add_history_ranks(user_item_features(past.user_listens, past.T)),
        user=user_features(past.user_listens, past.T),
        reactions=reactions,
        item_reactions=item_reactions,
        pop_items=popular_items(items, POP_WINDOW, POP_LIST_SIZE),
    )
