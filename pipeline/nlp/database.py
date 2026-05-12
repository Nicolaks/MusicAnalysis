"""
database.py
===========
Tout ce qui touche à DuckDB : initialisation des tables, chargement des
pistes à analyser, upsert générique.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from pipeline.nlp.config import ALL_DDL, DB_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Ouvre la connexion et crée les tables si elles n'existent pas."""
    con = duckdb.connect(str(db_path))
    for ddl in ALL_DDL:
        con.execute(ddl)
    return con


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_tracks_to_analyze(
    db_path: Path,
    artists: list[str],
    rerun: bool,
) -> list[dict]:
    """
    Retourne la liste des pistes à analyser depuis tracks_flat.
    Filtre par artiste et/ou exclut celles déjà en BDD selon `rerun`.
    """
    con = init_db(db_path)

    artist_filter = ""
    params: list = []
    
    if artists:
        normalized_artists = [
            (
                a.lower()
                .replace("'", "")
                .replace("’", "")
                .replace("-", " ")
                .strip()
            )
            for a in artists
        ]
        placeholders = ", ".join("?" * len(normalized_artists))
        artist_filter = f"""
            AND REPLACE(
                    REPLACE(
                        REPLACE(LOWER(tf.artist_name), '''', ''),
                    '’', ''),
                '-', ' '
            ) IN ({placeholders})
        """

    params = normalized_artists
    exists_filter = "" if rerun else """
        AND NOT EXISTS (
            SELECT 1 FROM tracks_analysis ta
            WHERE ta.track_id = tf.track_id AND ta.artist_id = tf.artist_id
        )"""

    rows = con.execute(f"""
        SELECT tf.track_id, tf.artist_id, tf.album_id,
               tf.track_name, tf.artist_name, tf.album_name,
               tf.album_release_year, tf.lyrics,
               id.isrc, NULL AS artist_isrc
        FROM tracks_flat tf
        LEFT JOIN isrc_data id ON id.track_id = tf.track_id
        WHERE tf.lyrics IS NOT NULL
        {artist_filter}
        {exists_filter}
        ORDER BY tf.artist_id, tf.album_id, tf.track_id
    """, params).fetchall()

    con.close()
    logger.info("%d pistes à analyser", len(rows))

    return [
        {
            "track_id": r[0], "artist_id": r[1], "album_id": r[2],
            "track_name": r[3], "artist_name": r[4], "album_name": r[5],
            "release_year": r[6], "lyrics": r[7],
            "isrc": r[8], "artist_isrc": r[9],
        }
        for r in rows
    ]


def load_table(
    db_path: Path,
    table: str,
    artists: list[str],
) -> list[dict]:
    """Charge toutes les lignes d'une table d'analyse, filtré par artiste."""
    con = init_db(db_path)
    artist_filter = ""
    params: list = []
    if artists:
        artist_filter = "WHERE LOWER(artist_name) IN (%s)" % ", ".join("?" * len(artists))
        params = [a.lower() for a in artists]

    rows = con.execute(f"SELECT * FROM {table} {artist_filter}", params).fetchall()
    cols = [d[0] for d in con.description]
    con.close()
    return [dict(zip(cols, row)) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Upsert
# ─────────────────────────────────────────────────────────────────────────────

def upsert(
    con: duckdb.DuckDBPyConnection,
    table: str,
    rows: list[dict],
) -> tuple[int, int]:
    """
    INSERT OR REPLACE générique.
    Retourne (nb_ok, nb_erreurs).
    """
    if not rows:
        return 0, 0

    cols = list(rows[0].keys())
    ph   = ", ".join(["?"] * len(cols))
    sql  = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({ph})"

    ok = bad = 0
    for row in rows:
        try:
            con.execute(sql, list(row.values()))
            ok += 1
        except Exception as e:
            label = row.get("track_name") or row.get("album_name") or row.get("artist_name")
            logger.error("INSERT %s [%s] : %s", table, label, e)
            bad += 1

    return ok, bad


def flush(db_path: Path, table: str, rows: list[dict]) -> None:
    """Ouvre une connexion, upsert, ferme. Raccourci pour les flush intermédiaires."""
    con = init_db(db_path)
    ok, bad = upsert(con, table, rows)
    con.close()
    if bad:
        logger.warning("flush %s — %d ok | %d erreurs", table, ok, bad)
