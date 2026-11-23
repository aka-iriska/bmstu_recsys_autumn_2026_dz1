from __future__ import annotations

import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RECSYS_DATA", ROOT / "data"))
OUT_DIR = Path(os.environ.get("RECSYS_OUT", ROOT / "output"))

DAY = 86400  # timestamp число секунд в сутках
SEED = 42
LISTEN, LIKE, UNLIKE, DISLIKE, UNDISLIKE = "listen", "like", "unlike", "dislike", "undislike"
# порядок категорий задаёт порядок событий с одинаковым timestamp при сортировке
EVENT_TYPE = pl.Enum([LISTEN, LIKE, UNLIKE, DISLIKE, UNDISLIKE])
