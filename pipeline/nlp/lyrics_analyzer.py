"""
lyrics_analyzer.py
==================
Point d'entrée du pipeline NLP des paroles — 3 niveaux d'analyse :

  1. tracks_analysis  → métriques par chanson
  2. albums_analysis  → agrégation + métriques propres à l'album
  3. artists_analysis → agrégation + métriques propres à l'artiste

Usage :
    python lyrics_analyzer.py                        # tout le corpus
    python lyrics_analyzer.py "PLK" "Nekfeu"         # artiste(s) ciblé(s)
    python lyrics_analyzer.py --rerun                # ré-analyse même si déjà en BDD
    python lyrics_analyzer.py --level track          # niveau unique (track|album|artist)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.nlp.config import DB_PATH, ensure_nltk
from pipeline.nlp.database import flush, init_db, load_table, load_tracks_to_analyze, upsert
from pipeline.nlp.helpers import clean_lyrics
from pipeline.nlp.models import build_lda
from pipeline.nlp.analyzers.track import analyze_track
from pipeline.nlp.analyzers.album import aggregate_album
from pipeline.nlp.analyzers.artist import aggregate_artist

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/lyrics_analyzer.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(
    artists: list[str] = None,
    db_path: Path = DB_PATH,
    rerun: bool = False,
    level: str = "all",
) -> None:
    ensure_nltk()
    artists = artists or []

    logger.info("=" * 60)
    logger.info(
        "Lyrics Analyzer — niveaux : %s — artistes : %s",
        level, ", ".join(artists) if artists else "tous",
    )
    logger.info("=" * 60)

    tracks_meta = load_tracks_to_analyze(db_path, artists, rerun)
    if not tracks_meta:
        logger.info("Rien à analyser.")
        return

    # LDA entraîné sur le corpus complet avant les analyses
    logger.info("Entraînement LDA sur %d documents…", len(tracks_meta))
    corpus_texts = [clean_lyrics(m["lyrics"]) for m in tracks_meta]
    lda_model    = build_lda(corpus_texts)

    # ── NIVEAU TRACK ──────────────────────────────────────────────────────────
    track_results: list[dict] = []

    if level in ("track", "all"):
        track_results = _run_tracks(tracks_meta, lda_model, db_path)

    # Si on saute le niveau track, on charge depuis la BDD
    if level in ("album", "artist") and not track_results:
        logger.info("Chargement tracks_analysis depuis BDD…")
        track_results = load_table(db_path, "tracks_analysis", artists)
        logger.info("%d tracks chargées", len(track_results))

    # ── NIVEAU ALBUM ──────────────────────────────────────────────────────────
    album_results: list[dict] = []

    if level in ("album", "all"):
        album_results = _run_albums(tracks_meta, track_results, lda_model, db_path)

    # ── NIVEAU ARTIST ─────────────────────────────────────────────────────────
    if level in ("artist", "all"):
        # Charge les albums depuis la BDD si on a sauté le niveau album
        if not album_results:
            album_results = load_table(db_path, "albums_analysis", artists)

        _run_artists(tracks_meta, track_results, album_results, lda_model, db_path)

    logger.info("=" * 60)
    logger.info("Pipeline terminé — base : %s", db_path.resolve())
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Sous-routines par niveau
# ─────────────────────────────────────────────────────────────────────────────

def _run_tracks(
    tracks_meta: list[dict],
    lda_model,
    db_path: Path,
) -> list[dict]:
    """Analyse chaque piste et écrit en BDD toutes les 50 pistes."""
    results: list[dict] = []
    total = len(tracks_meta)

    for i, m in enumerate(tracks_meta, 1):
        logger.info("[Track %d/%d] %s — %s", i, total, m["artist_name"], m["track_name"])
        try:
            row = analyze_track(
                track_id=m["track_id"], artist_id=m["artist_id"], album_id=m["album_id"],
                track_name=m["track_name"], artist_name=m["artist_name"],
                album_name=m["album_name"], raw_lyrics=m["lyrics"],
                isrc=m["isrc"],
                lda_model=lda_model,
            )
            results.append(row)
        except Exception as e:
            logger.error("❌ Track [%s] : %s", m["track_name"], e, exc_info=True)

        # Flush intermédiaire tous les 50
        if len(results) % 50 == 0 and results:
            flush(db_path, "tracks_analysis", results[-50:])

    # Flush du reste
    remainder = len(results) % 50
    if remainder:
        flush(db_path, "tracks_analysis", results[-remainder:])
    elif results and len(results) < 50:
        flush(db_path, "tracks_analysis", results)

    logger.info("✅ tracks_analysis : %d analysées", len(results))
    return results


def _run_albums(
    tracks_meta: list[dict],
    track_results: list[dict],
    lda_model,
    db_path: Path,
) -> list[dict]:
    """Agrège les métriques par album et écrit en BDD."""
    albums_map: dict[tuple, list] = {}
    for tr in tracks_meta:
        if tr.get("album_id"):
            albums_map.setdefault((tr["album_id"], tr["artist_id"]), []).append(tr)

    results: list[dict] = []
    for (album_id, artist_id), album_tracks in albums_map.items():
        track_ids = {t["track_id"] for t in album_tracks}
        analyzed  = [t for t in track_results if t.get("track_id") in track_ids]
        if not analyzed:
            continue

        sample = album_tracks[0]
        logger.info(
            "[Album] %s — %s (%d titres)",
            sample["artist_name"], sample["album_name"], len(analyzed),
        )
        try:
            row = aggregate_album(
                album_id=album_id, artist_id=artist_id,
                album_name=sample["album_name"], artist_name=sample["artist_name"],
                release_year=sample.get("release_year"),
                track_rows=analyzed,
                raw_lyrics_list=[t["lyrics"] for t in album_tracks],
                lda_model=lda_model,
            )
            results.append(row)
        except Exception as e:
            logger.error("❌ Album [%s] : %s", sample.get("album_name"), e, exc_info=True)

    con = init_db(db_path)
    ok, bad = upsert(con, "albums_analysis", results)
    con.close()
    logger.info("✅ albums_analysis : %d insérées | %d ignorées", ok, bad)
    return results


def _run_artists(
    tracks_meta: list[dict],
    track_results: list[dict],
    album_results: list[dict],
    lda_model,
    db_path: Path,
) -> None:
    """Agrège les métriques par artiste et écrit en BDD."""
    artists_map: dict[int, list] = {}
    for tr in tracks_meta:
        artists_map.setdefault(tr["artist_id"], []).append(tr)

    results: list[dict] = []
    for artist_id, artist_tracks in artists_map.items():
        track_ids  = {t["track_id"] for t in artist_tracks}
        analyzed   = [t for t in track_results if t.get("track_id") in track_ids]
        albums_for = [a for a in album_results if a.get("artist_id") == artist_id]
        sample     = artist_tracks[0]

        logger.info(
            "[Artist] %s — %d titres | %d albums",
            sample["artist_name"], len(analyzed), len(albums_for),
        )
        try:
            row = aggregate_artist(
                artist_id=artist_id,
                artist_name=sample["artist_name"],
                track_rows=analyzed,
                album_rows=albums_for,
                raw_lyrics_list=[t["lyrics"] for t in artist_tracks],
                lda_model=lda_model,
            )
            results.append(row)
        except Exception as e:
            logger.error("❌ Artist [%s] : %s", sample.get("artist_name"), e, exc_info=True)

    con = init_db(db_path)
    ok, bad = upsert(con, "artists_analysis", results)
    con.close()
    logger.info("✅ artists_analysis : %d insérées | %d ignorées", ok, bad)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline NLP lyrics — 3 niveaux.")
    parser.add_argument("artists", nargs="*", help="Artiste(s) ciblé(s) (vide = tous)")
    parser.add_argument("--db",    default=str(DB_PATH), help="Chemin DuckDB")
    parser.add_argument("--rerun", action="store_true",  help="Ré-analyse même si déjà en BDD")
    parser.add_argument(
        "--level", default="all",
        choices=["track", "album", "artist", "all"],
        help="Niveau d'analyse à exécuter (défaut : all)",
    )
    args = parser.parse_args()

    run(
        artists=args.artists or [],
        db_path=Path(args.db),
        rerun=args.rerun,
        level=args.level,
    )
