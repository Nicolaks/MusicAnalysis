import time
import logging
import duckdb
import requests

from pathlib import Path

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
    deezer_preview_url VARCHAR
);
"""

def _fetch_deezer(track_name: str, artist_name: str) -> dict | None:
    try:
        resp = requests.get(
            f"{DEEZER_BASE}/search",
            params={"q": f'track:"{track_name}" artist:"{artist_name}"', "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            hit = data[0]
            return {
                "isrc":               hit.get("isrc"),
                "deezer_track_id":    hit.get("id"),
                "deezer_preview_url": hit.get("preview"),
            }
    except requests.RequestException as e:
        logger.warning("Deezer erreur [%s - %s] : %s", artist_name, track_name, e)
    return None

def run(db_path: Path = DB_PATH) -> None:
    logger.info("=" * 60)
    logger.info("Démarrage enrichissement ISRC")
    logger.info("=" * 60)

    con = duckdb.connect(str(db_path))
    con.execute(DDL_ISRC)

    # Tracks déjà enrichis → skip
    already = con.execute("SELECT COUNT(*) FROM isrc_data WHERE isrc IS NOT NULL").fetchone()[0]
    logger.info("Cache isrc_data : %d track(s) déjà enrichis", already)

    # Récupère tous les tracks de tracks_flat pas encore dans isrc_data
    rows = con.execute("""
        SELECT track_id, track_name, artist_name, album_name, album_release_year
        FROM tracks_flat
        WHERE track_id NOT IN (SELECT track_id FROM isrc_data)
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
                "INSERT OR REPLACE INTO isrc_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    track_id,
                    track_name,
                    artist_name,
                    album_name,
                    release_year,
                    result["isrc"],
                    result["deezer_track_id"],
                    result["deezer_preview_url"],
                ],
            )
            enriched += 1
            logger.info("    ✅ ISRC : %s | Deezer id : %s", result["isrc"], result["deezer_track_id"])
        else:
            # Insère quand même une ligne avec isrc NULL pour ne pas retraiter à chaque fois
            con.execute(
                "INSERT OR REPLACE INTO isrc_data VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                [track_id, track_name, artist_name, album_name, release_year],
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
