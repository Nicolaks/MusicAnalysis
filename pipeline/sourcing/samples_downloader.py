"""
samples_downloader.py
=====================
Crée une table samples_index avec tous les titres ayant un extrait Deezer,
puis télécharge les fichiers MP3 dans data/samples/.

Les URLs Deezer sont signées et expirent rapidement — on les rafraîchit
juste avant chaque téléchargement via l'API Deezer.

Table samples_index :
    track_id, artist_name, album_name, track_name,
    preview_url, file_path, downloaded (YES / NO / ERROR)
"""

import time
import logging
import duckdb
import requests

from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH      = Path("data/warehouse.duckdb")
SAMPLES_DIR  = Path("data/samples")
DEEZER_BASE  = "https://api.deezer.com"
REQUEST_DELAY = 0.3

DDL_SAMPLES = """
CREATE TABLE IF NOT EXISTS samples_index (
    track_id     INTEGER PRIMARY KEY,
    artist_name  VARCHAR,
    album_name   VARCHAR,
    track_name   VARCHAR,
    preview_url  VARCHAR,
    file_path    VARCHAR,
    downloaded   VARCHAR DEFAULT 'NO'
);
"""

# ─────────────────────────────────────────────
# Init table
# ─────────────────────────────────────────────

def _init_table(con: duckdb.DuckDBPyConnection) -> None:
    """Crée la table et insère les tracks ayant un preview_url."""
    con.execute(DDL_SAMPLES)

    inserted = con.execute("""
        INSERT OR IGNORE INTO samples_index (track_id, artist_name, album_name, track_name, preview_url, file_path, downloaded)
        SELECT
            t.track_id,
            t.artist_name,
            t.album_name,
            t.track_name,
            i.deezer_preview_url,
            NULL,
            'NO'
        FROM tracks_flat t
        JOIN isrc_data i ON t.track_id = i.track_id
        WHERE i.deezer_preview_url IS NOT NULL
          AND t.track_id NOT IN (SELECT track_id FROM samples_index)
    """).rowcount

    total = con.execute("SELECT COUNT(*) FROM samples_index").fetchone()[0]
    logger.info("samples_index : %d tracks total | %d nouveaux ajoutés", total, inserted)


# ─────────────────────────────────────────────
# Refresh URL Deezer
# ─────────────────────────────────────────────

def _refresh_preview_url(deezer_track_id: int) -> str | None:
    """
    Récupère une URL de preview fraîche depuis l'API Deezer.
    Les URLs sont signées avec expiration — impossible de réutiliser
    une URL stockée en base.
    """
    try:
        resp = requests.get(
            f"{DEEZER_BASE}/track/{deezer_track_id}",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("preview")
    except requests.RequestException as e:
        logger.warning("    Refresh preview URL échoué (id=%s) : %s", deezer_track_id, e)
        return None


# ─────────────────────────────────────────────
# Téléchargement
# ─────────────────────────────────────────────

def _safe_filename(artist: str, track: str, track_id: int) -> str:
    """Génère un nom de fichier sûr depuis artiste + titre + id."""
    def clean(s: str) -> str:
        return "".join(c if c.isalnum() or c in " -_" else "_" for c in s).strip()
    return f"{clean(artist)} - {clean(track)} [{track_id}].mp3"


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """Télécharge un fichier MP3 avec retry. Retourne True si succès."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=20, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning("    Tentative %d/%d échouée : %s — attente %ds",
                           attempt + 1, retries, e, wait)
            time.sleep(wait)
    return False


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

def run(db_path: Path = DB_PATH, samples_dir: Path = SAMPLES_DIR) -> None:
    logger.info("=" * 60)
    logger.info("Démarrage téléchargement samples")
    logger.info("=" * 60)

    samples_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    _init_table(con)

    # Récupère aussi deezer_track_id pour le refresh URL
    rows = con.execute("""
        SELECT s.track_id, s.artist_name, s.album_name, s.track_name,
               i.deezer_track_id
        FROM samples_index s
        JOIN isrc_data i ON s.track_id = i.track_id
        WHERE s.downloaded IN ('NO', 'ERROR')
        ORDER BY s.artist_name, s.track_name
    """).fetchall()

    total      = len(rows)
    downloaded = 0
    errors     = 0
    skipped    = 0

    logger.info("%d sample(s) à télécharger", total)

    for i, (track_id, artist_name, album_name, track_name, deezer_track_id) in enumerate(rows, start=1):
        logger.info("[%d/%d] %s — %s", i, total, artist_name, track_name)

        # Destination sur disque
        artist_dir = samples_dir / "".join(
            c if c.isalnum() or c in " -_" else "_" for c in artist_name
        ).strip()
        filename = _safe_filename(artist_name, track_name, track_id)
        dest     = artist_dir / filename

        # Déjà téléchargé sur disque
        if dest.exists():
            logger.info("    ⏭ Déjà présent sur disque : %s", dest.name)
            con.execute(
                "UPDATE samples_index SET downloaded = 'YES', file_path = ? WHERE track_id = ?",
                [str(dest), track_id],
            )
            skipped += 1
            continue

        # Refresh URL fraîche (les URLs Deezer expirent)
        preview_url = _refresh_preview_url(deezer_track_id)
        if not preview_url:
            logger.error("    ❌ URL de preview introuvable pour deezer_track_id=%s", deezer_track_id)
            con.execute(
                "UPDATE samples_index SET downloaded = 'ERROR' WHERE track_id = ?",
                [track_id],
            )
            errors += 1
            time.sleep(REQUEST_DELAY)
            continue

        time.sleep(REQUEST_DELAY)

        # Téléchargement
        success = _download(preview_url, dest)

        if success:
            con.execute(
                "UPDATE samples_index SET downloaded = 'YES', file_path = ? WHERE track_id = ?",
                [str(dest), track_id],
            )
            downloaded += 1
            logger.info("    ✅ %s", dest.name)
        else:
            con.execute(
                "UPDATE samples_index SET downloaded = 'ERROR' WHERE track_id = ?",
                [track_id],
            )
            errors += 1
            logger.error("    ❌ Échec téléchargement : %s", track_name)

    con.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Terminé : %d téléchargés | %d déjà présents | %d erreurs", downloaded, skipped, errors)
    logger.info("  Dossier : %s", samples_dir.resolve())
    logger.info("=" * 60)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/samples_downloader.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    Path("logs").mkdir(exist_ok=True)
    run(db_path=DB_PATH, samples_dir=SAMPLES_DIR)