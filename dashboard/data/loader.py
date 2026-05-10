from __future__ import annotations

import duckdb
import pandas as pd
from functools import lru_cache
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DB_PATH


def _con(db_path: Path = DB_PATH):
    return duckdb.connect(str(db_path), read_only=True)


def _table_exists(con, table: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_artists(db_path: Path = DB_PATH) -> list[str]:
    try:
        con = _con(db_path)
        rows = con.execute(
            "SELECT DISTINCT artist_name FROM artists_analysis ORDER BY artist_name"
        ).fetchall()
        con.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []

def get_artist_url(artist_name: str, db_path: Path = DB_PATH) -> str | None:
    try:
        con = _con(db_path)
        df = con.execute(
            "SELECT artist_image_url FROM tracks_flat WHERE artist_name = ?",
            [artist_name]
        ).df()
        con.close()

        if df.empty:
            return None

        return df.iloc[0]["artist_image_url"]

    except Exception:
        return None

def get_artist(artist_name: str, db_path: Path = DB_PATH) -> pd.Series:
    try:
        con = _con(db_path)
        df = con.execute(
            "SELECT * FROM artists_analysis WHERE artist_name = ?", [artist_name]
        ).df()
        con.close()
        return df.iloc[0] if not df.empty else pd.Series(dtype=object)
    except Exception:
        return pd.Series(dtype=object)


def get_albums(artist_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    try:
        con = _con(db_path)
        df = con.execute("""
            SELECT * FROM albums_analysis
            WHERE artist_name = ?
            ORDER BY release_year NULLS LAST, album_name
        """, [artist_name]).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_tracks(artist_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    try:
        con = _con(db_path)
        has_kworb   = _table_exists(con, "kworb_streams")
        has_ranking = _table_exists(con, "ranking_data")

        joins, extra_cols = "", "ta.*"
        if has_kworb:
            joins      += " LEFT JOIN kworb_streams ks ON ks.track_id = ta.track_id"
            extra_cols += ", ks.streams, ks.daily_streams"
        if has_ranking:
            joins      += " LEFT JOIN ranking_data rd ON rd.track_id = ta.track_id"
            extra_cols += ", rd.spotify_total_streams, rd.apple_total, rd.youtube_views"

        df = con.execute(f"""
            SELECT {extra_cols}
            FROM tracks_analysis ta
            {joins}
            WHERE ta.artist_name = ?
        """, [artist_name]).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_track(track_id: int, db_path: Path = DB_PATH) -> pd.Series:
    try:
        con = _con(db_path)
        df = con.execute(
            "SELECT * FROM tracks_analysis WHERE track_id = ?", [track_id]
        ).df()
        con.close()
        return df.iloc[0] if not df.empty else pd.Series(dtype=object)
    except Exception:
        return pd.Series(dtype=object)


def get_all_tracks(db_path: Path = DB_PATH) -> pd.DataFrame:
    try:
        con = _con(db_path)
        has_kworb = _table_exists(con, "kworb_streams")
        extra = ", ks.streams" if has_kworb else ""
        join  = "LEFT JOIN kworb_streams ks ON ks.track_id = ta.track_id" if has_kworb else ""
        df = con.execute(f"""
            SELECT ta.* {extra}
            FROM tracks_analysis ta {join}
        """).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_audio_features(artist_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    try:
        con = _con(db_path)
        if not _table_exists(con, "audio_features_local"):
            con.close()
            return pd.DataFrame()
        df = con.execute("""
            SELECT afl.*, ta.ttr, ta.rhyme_density, ta.semantic_density,
                   ta.sentiment_negative, ta.sentiment_positive,
                   ta.lexical_diversity, ta.hapax_ratio,
                   ta.avg_word_length
            FROM audio_features_local afl
            JOIN tracks_analysis ta ON ta.track_id = afl.track_id
            WHERE afl.artist_name = ?
        """, [artist_name]).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_all_audio_nlp(db_path: Path = DB_PATH) -> pd.DataFrame:
    try:
        con = _con(db_path)
        if not _table_exists(con, "audio_features_local"):
            con.close()
            return pd.DataFrame()
        df = con.execute("""
            SELECT afl.tempo, afl.beat_strength, afl.rms_mean,
                   afl.dynamic_range, afl.spectral_centroid,
                   afl.brightness, afl.warmth, afl.roughness,
                   ta.ttr, ta.rhyme_density, ta.semantic_density,
                   ta.sentiment_negative, ta.avg_word_length,
                   ta.lexical_diversity, ta.hapax_ratio
            FROM audio_features_local afl
            JOIN tracks_analysis ta ON ta.track_id = afl.track_id
        """).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_artists_comparison(artist_names: list[str], db_path: Path = DB_PATH) -> pd.DataFrame:
    if not artist_names:
        return pd.DataFrame()
    try:
        con = _con(db_path)
        placeholders = ", ".join(["?"] * len(artist_names))
        has_kworb = _table_exists(con, "kworb_streams")
        stream_join = ""
        stream_col  = "NULL AS total_streams"
        if has_kworb:
            stream_join = """
                LEFT JOIN tracks_analysis ta2 ON ta2.artist_id = aa.artist_id
                LEFT JOIN kworb_streams ks    ON ks.track_id   = ta2.track_id
            """
            stream_col = "SUM(DISTINCT ks.streams) AS total_streams"

        df = con.execute(f"""
            SELECT aa.*, {stream_col}
            FROM artists_analysis aa
            {stream_join}
            WHERE aa.artist_name IN ({placeholders})
            GROUP BY aa.artist_id
        """, artist_names).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_corpus_stats(db_path: Path = DB_PATH) -> dict:
    try:
        con = _con(db_path)
        r = {}
        r["total_artists"] = con.execute("SELECT COUNT(*) FROM artists_analysis").fetchone()[0]
        r["total_albums"]  = con.execute("SELECT COUNT(*) FROM albums_analysis").fetchone()[0]
        r["total_tracks"]  = con.execute("SELECT COUNT(*) FROM tracks_analysis").fetchone()[0]
        r["total_words"]   = con.execute(
            "SELECT COALESCE(SUM(word_count),0) FROM tracks_analysis"
        ).fetchone()[0]
        con.close()
        return r
    except Exception:
        return {"total_artists": 0, "total_albums": 0, "total_tracks": 0, "total_words": 0}
