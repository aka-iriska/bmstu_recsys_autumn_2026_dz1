from __future__ import annotations

import time

import polars as pl


def log(msg: str) -> None:
    """Печатает сообщение с текущим временем."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rank_in_user(col: str) -> pl.Expr:
    """
    Место трека среди треков того же пользователя при сортировке по убыванию col (1 - наибольшее значение).

    Пример: у пользователя три трека с ui_cnt = 5, 9, 2 -> номера 2, 1, 3. При равных значениях
    раньше получает номер та строка, что выше в таблице.
    """
    return pl.col(col).rank("ordinal", descending=True).over("uid").cast(pl.Int32)
