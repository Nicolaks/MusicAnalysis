from __future__ import annotations

import json
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMOTION_LABELS, RADAR_KEYS


def safe_json(val) -> list | dict | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return json.loads(val)
    except Exception:
        return None


def parse_top_words(val) -> list[str]:
    r = safe_json(val)
    return r if isinstance(r, list) else []


def parse_style_signature(val) -> dict:
    r = safe_json(val)
    return r if isinstance(r, dict) else {}


def parse_emotion_arc(val) -> list[str]:
    r = safe_json(val)
    return r if isinstance(r, list) else []


def parse_lda_distribution(val) -> dict[str, float]:
    r = safe_json(val)
    return {k: float(v) for k, v in r.items()} if isinstance(r, dict) else {}


def normalize_radar(values: dict, clip: float = 1.0) -> dict:
    vals = np.array(list(values.values()), dtype=float)
    mn, mx = np.nanmin(vals), np.nanmax(vals)
    if mx == mn:
        return {k: 0.5 for k in values}
    return {k: float(np.clip((v - mn) / (mx - mn), 0, clip)) for k, v in values.items()}


def albums_emotion_matrix(df_albums: pd.DataFrame) -> pd.DataFrame:
    if df_albums.empty or "avg_emotion_scores" not in df_albums.columns:
        return pd.DataFrame()

    records = []
    for _, row in df_albums.iterrows():
        try:
            scores = json.loads(row["avg_emotion_scores"])
        except (TypeError, json.JSONDecodeError):
            scores = {}
        scores["album_name"]   = row["album_name"]
        scores["release_year"] = row.get("release_year")
        records.append(scores)

    sub = pd.DataFrame(records)
    sub = sub.sort_values("release_year", na_position="last")
    sub = sub.set_index("album_name").drop(columns=["release_year"], errors="ignore")
    return sub


def safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except Exception:
        return default


def streams_label(n) -> str:
    try:
        n = int(n)
    except Exception:
        return "—"
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def build_radar_values(row: pd.Series, keys: list[str]) -> dict:
    return {k: safe_float(row.get(k, 0)) for k in keys}

def normalize_fk_grade(score: float, min_score: float, max_score: float) -> float:
    if max_score == min_score:
        return 0.5
    clipped = max(min_score, min(max_score, score))
    normalized = (clipped - min_score) / (max_score - min_score)
    return round(max(0.05, 1 - normalized), 3)  # plancher à 0.05
