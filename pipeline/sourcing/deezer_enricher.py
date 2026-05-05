import time
import logging
import duckdb
import requests
import re

from pathlib import Path
from unidecode import unidecode
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

DB_PATH       = Path("data/warehouse.duckdb")
DEEZER_BASE   = "https://api.deezer.com"
REQUEST_DELAY = 0.5

DDL_ISRC = """
CREATE TABLE IF NOT EXISTS isrc_data (
    track_id          INTEGER PRIMARY KEY,
    track_name        VARCHAR,
    artist_name       VARCHAR,
    album_name        VARCHAR,
    release_year      INTEGER,
    isrc              VARCHAR,
    deezer_track_id   BIGINT,
    deezer_preview_url VARCHAR,
    match_strategy     VARCHAR,
    deezer_duration     INTEGER,
    deezer_rank         BIGINT,
    deezer_explicit     BOOLEAN
);
"""

def _normalize(text: str) -> str:
    text = unidecode(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _clean_title(title: str) -> str:
    
    title = unidecode(title.lower())
    title = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", title)
    
    blacklist = [
        "freestyle", "remix", "version", "radio",
        "oklm", "booska", "planete rap", "colors",
        "live", "clip", "demo", "extrait", "inedit"
    ]
    
    for word in blacklist:
        title = re.sub(rf"\b{word}\b", "", title)
    
    title = re.sub(r"[^\w\s]", " ", title)

    return re.sub(r"\s+", " ", title).strip()

def _deezer_search(q: str, limit: int = 5) -> list[dict]:
    try:
        resp = requests.get(
            f"{DEEZER_BASE}/search",
            params={"q": q, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.RequestException as e:
        logger.warning("    Deezer erreur [q=%s] : %s", q, e)
        return []

def _extract(hit: dict, strategy: str) -> dict:
    return {
        "isrc":               hit.get("isrc"),
        "deezer_track_id":    hit.get("id"),
        "deezer_preview_url": hit.get("preview"),
        "match_strategy":     strategy,
        "deezer_duration":    hit.get("duration"),
        "deezer_rank":        hit.get("rank"),
        "deezer_explicit":    hit.get("explicit_lyrics"),
    }

def _artist_matches(hit: dict, artist_name: str) -> bool:
    deezer_artist = _normalize(hit.get("artist", {}).get("name", ""))
    target = _normalize(artist_name)
    return (
        target in deezer_artist or deezer_artist in target or fuzz.ratio(target, deezer_artist) > 95   
    )

def _fetch_deezer(track_name: str, artist_name: str) -> dict | None:

    results = _deezer_search(f'track:"{track_name}" artist:"{artist_name}"')
    for hit in results:
        if hit.get("isrc"):
            logger.info("    🎯 Match exact")
            return _extract(hit, "exact")

    q2 = f"{_normalize(track_name)} {_normalize(artist_name)}"
    results = _deezer_search(q2)
    for hit in results:
        if hit.get("isrc") and _artist_matches(hit, artist_name):
            logger.info("    🎯 Match souple (sans accents)")
            return _extract(hit, "souple")

    clean   = _clean_title(track_name)
    if clean != track_name:
        q3      = f'track:"{clean}" artist:"{artist_name}"'
        results = _deezer_search(q3)
        for hit in results:
            if hit.get("isrc") and _artist_matches(hit, artist_name):
                logger.info("    🎯 Match titre nettoyé")
                return _extract(hit, "clean_title")

    results = _deezer_search(_normalize(clean or track_name), limit=10)
    for hit in results:
        if hit.get("isrc") and _artist_matches(hit, artist_name):
            logger.info("    🎯 Match titre seul + vérif artiste")
            return _extract(hit, "titre_seul")
        
    results = _deezer_search(_normalize(track_name), limit=20)
    if results:
        def _score(hit: dict) -> float:
            title_score = fuzz.ratio(
                _normalize(hit.get("title", "")),
                _normalize(track_name)
            )
            artist_score = fuzz.ratio(
                _normalize(hit.get("artist", {}).get("name", "")),
                _normalize(artist_name)
            )
            return title_score * 0.5 + artist_score * 0.5
        best = max(results, key=_score)
        score = _score(best)
        logger.info("    🔍 Fuzzy best score : %.1f | titre : %s | artiste : %s", score, best.get("title", "?"), best.get("artist", {}).get("name", "?"))
        if score > 93 and best.get("isrc"):
            logger.info("    🎯 Match fuzzy (score=%.1f)", score)
            return _extract(best, f"fuzzy_{score:.0f}")
        else:
            logger.warning("    ❌ Fuzzy trop bas (%.1f) | meilleur candidat : %s — %s",
                           score,
                           best.get("artist", {}).get("name", "?"),
                           best.get("title", "?"))

    return None

def run(db_path: Path = DB_PATH) -> None:
    logger.info("=" * 60)
    logger.info("Démarrage enrichissement ISRC")
    logger.info("=" * 60)

    con = duckdb.connect(str(db_path))
    con.execute(DDL_ISRC)

    existing_cols = [r[0] for r in con.execute("DESCRIBE isrc_data").fetchall()]
    if "match_strategy" not in existing_cols:
        con.execute("ALTER TABLE isrc_data ADD COLUMN match_strategy VARCHAR")

    already = con.execute("SELECT COUNT(*) FROM isrc_data WHERE isrc IS NOT NULL").fetchone()[0]
    logger.info("Cache isrc_data : %d track(s) déjà enrichis", already)

    rows = con.execute("""
        SELECT track_id, track_name, artist_name, album_name, album_release_year
        FROM tracks_flat
        WHERE track_id NOT IN (
            SELECT track_id FROM isrc_data
        )
        ORDER BY artist_name, album_release_year, track_name
    """).fetchall()

    total    = len(rows)
    enriched = 0
    failed   = 0

    logger.info("%d titre(s) à enrichir", total)

    for i, (track_id, track_name, artist_name, album_name, release_year) in enumerate(rows, start=1):
        logger.info("[%d/%d] %s — %s", i, total, artist_name, track_name)

        result = _fetch_deezer(track_name, artist_name)

        if result and result.get("isrc"):
            con.execute(
                "INSERT OR REPLACE INTO isrc_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    track_id, track_name, artist_name, album_name, release_year,
                    result["isrc"], result["deezer_track_id"],
                    result["deezer_preview_url"], result["match_strategy"],
                    result["deezer_duration"], result["deezer_rank"], result["deezer_explicit"],
                ],
            )
            enriched += 1
            logger.info("    ✅ ISRC : %s | stratégie : %s", result["isrc"], result["match_strategy"])
        else:
            con.execute(
                "INSERT OR REPLACE INTO isrc_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [track_id, track_name, artist_name, album_name, release_year, None, None, None, None, None, None, None],
            )
            failed += 1
            logger.warning("    ❌ ISRC introuvable")

        time.sleep(REQUEST_DELAY)

    con.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Terminé : %d/%d ISRC récupérés | %d introuvables", enriched, total, failed)
    logger.info("=" * 60)

if __name__ == "__main__":
    import logging
    from pathlib import Path
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/isrc_enricher.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    Path("logs").mkdir(exist_ok=True)
    run(db_path=DB_PATH)
