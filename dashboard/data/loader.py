from __future__ import annotations

import duckdb
import pandas as pd
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
import json as _json
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
        row = con.execute(
            "SELECT artist_image_url FROM tracks_flat WHERE artist_name = ? LIMIT 1",
            [artist_name]
        ).fetchone()
        con.close()
        return row[0] if row else None
    except Exception:
            return None

def get_audio_radar(artist_name: str, db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne (artist_avg, corpus_avg) — chacun est un DataFrame 1 ligne
    avec les moyennes des features audio pour le radar chart.
    """
    cols = ", ".join([
        "AVG(tempo) as tempo",
        "AVG(beat_strength) as beat_strength",
        "AVG(brightness) as brightness",
        "AVG(warmth) as warmth",
        "AVG(roughness) as roughness",
        "AVG(onset_rate) as onset_rate",
    ])

    try:
        con = _con(db_path)
        if not _table_exists(con, "audio_features_local"):
            con.close()
            return pd.DataFrame(), pd.DataFrame()

        artist_avg = con.execute(f"""
            SELECT {cols}
            FROM audio_features_local
            WHERE artist_name = ?
        """, [artist_name]).df()

        corpus_avg = con.execute(f"""
            SELECT {cols}
            FROM audio_features_local
        """).df()

        con.close()
        return artist_avg, corpus_avg

    except Exception:
        return pd.DataFrame(), pd.DataFrame()

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
    
def get_streams_artist(artist_name: str, db_path: Path = DB_PATH) -> int:
    try:
        con = _con(db_path)
        if not _table_exists(con, "kworb_streams"):
            con.close()
            return None
        row = con.execute(
            "SELECT SUM(streams) FROM kworb_streams WHERE artist_name = ?",
            [artist_name]
        ).fetchone()
        con.close()
        return row[0] if row else 0
    except Exception as e:
        print(f"ERREUR: {e}")
        return 0

def get_albums_with_streams(artist_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    try:
        con = _con(db_path)
        has_kworb = _table_exists(con, "kworb_streams")

        if has_kworb:
            df = con.execute("""
                SELECT aa.*, SUM(ks.streams) AS total_streams
                FROM albums_analysis aa
                LEFT JOIN tracks_analysis ta ON ta.album_id = aa.album_id
                LEFT JOIN kworb_streams ks   ON ks.track_id = ta.track_id
                WHERE aa.artist_name = ?
                GROUP BY ALL
                ORDER BY aa.release_year NULLS LAST, aa.album_name
            """, [artist_name]).df()
        else:
            df = con.execute("""
                SELECT * FROM albums_analysis
                WHERE artist_name = ?
                ORDER BY release_year NULLS LAST, album_name
            """, [artist_name]).df()

        con.close()
        return df
    except Exception as e:
        print(f"ERREUR: {e}")
        return pd.DataFrame()

def get_tracks(artist_name: str | list[str], db_path: Path = DB_PATH) -> pd.DataFrame:
    try:
        con = _con(db_path)

        has_kworb = _table_exists(con, "kworb_streams")
        has_ranking = _table_exists(con, "ranking_data")

        joins = ""
        extra_cols = "ta.*"

        if has_kworb:
            joins += " LEFT JOIN kworb_streams ks ON ks.track_id = ta.track_id"
            extra_cols += ", ks.streams, ks.daily_streams"

        if has_ranking:
            joins += " LEFT JOIN ranking_data rd ON rd.track_id = ta.track_id"
            extra_cols += ", rd.spotify_total_streams, rd.apple_total, rd.youtube_views"

        # Gestion string ou liste
        if isinstance(artist_name, str):
            artist_names = [artist_name]
        else:
            artist_names = artist_name

        # Sécurité si liste vide
        if not artist_names:
            return pd.DataFrame()

        # Génère (?, ?, ?, ...)
        placeholders = ",".join(["?"] * len(artist_names))

        query = f"""
            SELECT {extra_cols}
            FROM tracks_analysis ta
            {joins}
            WHERE ta.artist_name IN ({placeholders})
        """

        df = con.execute(query, artist_names).df()

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
                LEFT JOIN kworb_streams ks     ON ks.track_id  = ta2.track_id
            """
            stream_col = "SUM(DISTINCT ks.streams) AS total_streams"

        df = con.execute(f"""
            SELECT aa.*, {stream_col}
            FROM artists_analysis aa
            {stream_join}
            WHERE aa.artist_name IN ({placeholders})
            GROUP BY ALL
        """, artist_names).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame()
    
def get_corpus_year_range(db_path: Path = DB_PATH) -> dict:
    try:
        con = _con(db_path)
        row = con.execute("""
            SELECT MIN(album_release_year), MAX(album_release_year)
            FROM tracks_flat
            WHERE album_release_year IS NOT NULL
            AND (is_canonical IS NULL OR is_canonical = TRUE)
        """).fetchone()
        con.close()
        return {"year_min": row[0], "year_max": row[1]}
    except Exception:
        return {"year_min": "—", "year_max": "—"}


def get_corpus_stats(db_path: Path = DB_PATH) -> dict:
    try:
        con = _con(db_path)
        r = {}

        # ── 1. Comptages globaux : 1 seule requête ──────────────────────
        counts = con.execute("""
            SELECT
                (SELECT COUNT(*) FROM artists_analysis) AS total_artists,
                (SELECT COUNT(*) FROM albums_analysis)  AS total_albums,
                (SELECT COUNT(*) FROM tracks_analysis)  AS total_tracks,
                (SELECT COALESCE(SUM(word_count), 0) FROM tracks_analysis) AS total_words
        """).fetchone()
        r["total_artists"] = counts[0]
        r["total_albums"]  = counts[1]
        r["total_tracks"]  = counts[2]
        r["total_words"]   = counts[3]

        # ── 2. Stats stylo : 1 seule requête, 1 seul scan ───────────────
        stylo_metrics = [
            "avg_pos_noun_ratio",
            "avg_pos_verb_ratio",
            "avg_pos_adj_ratio",
            "avg_pos_adv_ratio",
            "avg_pos_pron_ratio",
            "avg_pronoun_i_ratio",
            "avg_pronoun_we_ratio",
            "avg_pronoun_you_ratio",
            "avg_rhyme_density",
            "avg_syllables_line",
            "avg_repetition_ratio",
            "avg_flesch_reading_ease",
            "avg_flesch_kincaid_grade",
            "avg_word_length",
            "avg_semantic_density",
            "avg_lexical_diversity",
            "avg_hapax_ratio",
        ]

        # Génère MIN, MAX, AVG, MEDIAN pour chaque colonne en une passe
        select_parts = []
        for col in stylo_metrics:
            select_parts.append(f"""
                MIN({col})                                              AS {col}_min,
                MAX({col})                                              AS {col}_max,
                AVG({col})                                             AS {col}_avg,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col})    AS {col}_median
            """)
        stylo_sql = f"SELECT {', '.join(select_parts)} FROM artists_analysis"
        stylo_row = con.execute(stylo_sql).fetchone()

        # Réassemble dans r[] avec les mêmes clés qu'avant
        for i, col in enumerate(stylo_metrics):
            base = i * 4
            r[f"{col}_min"]    = stylo_row[base]     or 0
            r[f"{col}_max"]    = stylo_row[base + 1] or 1
            r[f"{col}_avg"]    = stylo_row[base + 2] or 0
            r[f"{col}_median"] = stylo_row[base + 3] or 0

        # ── 3. Champs lexicaux : inchangé (dépend de JSON applicatif) ───
        rows_lex = con.execute("""
            SELECT avg_lexical_field_scores
            FROM artists_analysis
            WHERE avg_lexical_field_scores IS NOT NULL
        """).fetchall()

        lex_totals = defaultdict(float)
        lex_count  = 0
        for (val,) in rows_lex:
            try:
                parsed = _json.loads(val) if isinstance(val, str) else val
                if parsed:
                    for k, v in parsed.items():
                        lex_totals[k] += float(v)
                    lex_count += 1
            except Exception:
                pass
        r["avg_lexical_fields"] = (
            {k: v / lex_count for k, v in lex_totals.items()} if lex_count else {}
        )

        con.close()
        return r

    except Exception as e:
        return {"total_artists": 0, "total_albums": 0, "total_tracks": 0, "total_words": 0}
      
def get_embeddings_all_artists(db_path=DB_PATH):
    con = _con(db_path)
    rows = con.execute("""
        SELECT artist_name, career_embedding_centroid 
        FROM artists_analysis 
        WHERE career_embedding_centroid IS NOT NULL
    """).fetchall()
    con.close()
    return rows