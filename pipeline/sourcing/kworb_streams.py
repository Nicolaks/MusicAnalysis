"""
kworb_streams.py
================
Récupère les streams Spotify depuis Kworb.net pour chaque artiste
de la BDD et les stocke dans une table kworb_streams.

Flux :
    artists de tracks_flat
        → search Spotify ID artiste (API Spotify)
        → scraping kworb.net/spotify/artist/{id}_songs.html
        → matching via Spotify track ID ou fuzzy titre
        → kworb_streams

Table kworb_streams :
    track_id, artist_name, track_name,
    spotify_track_id, streams, daily_streams,
    is_feature, last_updated

Prérequis :
    pip install requests beautifulsoup4 duckdb spotipy rapidfuzz python-dotenv
"""

import os
import re
import time
import logging
import duckdb
import requests
import spotipy

from pathlib import Path
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
DB_PATH       = Path("data/warehouse.duckdb")

SCRAPE_DELAY  = 2.0   # secondes entre scrapings Kworb
SEARCH_DELAY  = 0.3   # secondes entre appels API Spotify
FUZZY_THRESHOLD = 77

DDL_KWORB = """
CREATE TABLE IF NOT EXISTS kworb_streams (
    track_id         INTEGER PRIMARY KEY,
    artist_name      VARCHAR,
    track_name       VARCHAR,
    spotify_track_id VARCHAR,
    streams          BIGINT,
    daily_streams    BIGINT,
    is_feature       BOOLEAN,
    kworb_match_str  VARCHAR,
    match_score      FLOAT,
    last_updated     VARCHAR
);
"""


# ─────────────────────────────────────────────
# Client Spotify (pour résoudre l'ID artiste)
# ─────────────────────────────────────────────

def _get_spotify() -> spotipy.Spotify | None:
    if not CLIENT_ID or not CLIENT_SECRET:
        logger.warning("Credentials Spotify absents — résolution ID artiste désactivée")
        return None
    auth = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return spotipy.Spotify(auth_manager=auth, requests_timeout=10)


def _get_spotify_artist_id(sp: spotipy.Spotify, artist_name: str) -> str | None:
    """Résout le nom d'un artiste en ID Spotify."""
    try:
        results = sp.search(q=artist_name, type="artist", limit=1)
        items   = results["artists"]["items"]
        if items:
            artist_id = items[0]["id"]
            logger.info("    Spotify ID artiste : %s → %s", artist_name, artist_id)
            return artist_id
    except Exception as e:
        logger.warning(" ❌  Impossible de résoudre l'ID Spotify pour %s : %s", artist_name, e)
    return None


# ─────────────────────────────────────────────
# Scraping Kworb
# ─────────────────────────────────────────────

def _scrape_kworb(spotify_artist_id: str, artist_name: str) -> list[dict]:
    """
    Scrape la page songs Kworb d'un artiste.
    Retourne une liste de dicts :
        { title, spotify_track_id, streams, daily, is_feature, last_updated }
    """
    url = f"https://kworb.net/spotify/artist/{spotify_artist_id}_songs.html"
    logger.info("    Scraping Kworb : %s", url)

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("    Kworb inaccessible pour %s : %s", artist_name, e)
        return []

    soup    = BeautifulSoup(resp.text, "html.parser")
    songs   = []

    # Récupère la date de mise à jour
    last_updated = None
    for text in soup.stripped_strings:
        if re.match(r"\d{4}/\d{2}/\d{2}", text):
            last_updated = text.strip()
            break

    # Parse le tableau
    for row in soup.select("table tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        link = cells[0].find("a", href=True)
        if not link:
            continue

        # Spotify track ID depuis le href
        href  = link.get("href", "")
        match = re.search(r"spotify\.com/track/([A-Za-z0-9]+)", href)
        if not match:
            continue

        spotify_track_id = match.group(1)
        raw_title        = link.get_text(strip=True)
        is_feature       = raw_title.startswith("*")
        title            = raw_title.lstrip("* ").strip()

        def _parse_int(val: str) -> int | None:
            try:
                return int(val.replace(",", "").replace(".", "").strip())
            except ValueError:
                return None

        streams = _parse_int(cells[1].get_text(strip=True))
        daily   = _parse_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None

        songs.append({
            "title":            title,
            "spotify_track_id": spotify_track_id,
            "streams":          streams,
            "daily":            daily,
            "is_feature":       is_feature,
            "last_updated":     last_updated,
        })

    logger.info("  ✅   %d titres scrapés sur Kworb", len(songs))
    return songs


# ─────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    try:
        from unidecode import unidecode
        text = unidecode(text)
    except ImportError:
        pass
    text = re.sub(r"\(feat\.?.*?\)|\[.*?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _match_track(
    track_name: str,
    kworb_songs: list[dict],
) -> dict | None:
    """
    Tente de matcher un track de la BDD avec un titre Kworb.
    Retourne le dict Kworb matché ou None.
    """
    if not kworb_songs:
        return None

    titles = [_normalize(s["title"]) for s in kworb_songs]
    query  = _normalize(track_name)

    result = process.extractOne(
        query,
        titles,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )
    if result:
        _, score, idx = result
        song = kworb_songs[idx].copy()
        song["match_score"] = float(score)
        return song
    
    # Log du meilleur candidat raté pour audit
    best = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)
    if best:
        logger.debug("  ❌ %s | meilleur candidat : %s (%.1f)", query, best[0], best[1])

    return None


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

def run(db_path: Path = DB_PATH) -> None:
    logger.info("=" * 60)
    logger.info("Démarrage enrichissement streams Kworb")
    logger.info("=" * 60)

    sp  = _get_spotify()
    con = duckdb.connect(str(db_path))
    con.execute(DDL_KWORB)

    already = con.execute("SELECT COUNT(*) FROM kworb_streams").fetchone()[0]
    logger.info("Cache kworb_streams : %d tracks déjà enrichis", already)

    # Récupère les artistes distincts avec leurs tracks
    artists = con.execute("""
        SELECT DISTINCT artist_name
        FROM tracks_flat
        WHERE track_id NOT IN (SELECT track_id FROM kworb_streams)
        ORDER BY artist_name
    """).fetchall()

    total_artists  = len(artists)
    total_matched  = 0
    total_failed   = 0

    logger.info("%d artiste(s) à traiter", total_artists)

    for i, (artist_name,) in enumerate(artists, start=1):
        logger.info("")
        logger.info("── Artiste %d/%d : %s", i, total_artists, artist_name)

        # Tracks de cet artiste pas encore dans kworb_streams
        tracks = con.execute("""
            SELECT track_id, track_name
            FROM tracks_flat
            WHERE artist_name = ?
              AND track_id NOT IN (SELECT track_id FROM kworb_streams)
        """, [artist_name]).fetchall()

        if not tracks:
            logger.info("    Tous les tracks déjà enrichis")
            continue

        # Résolution ID Spotify artiste
        spotify_artist_id = None
        if sp:
            spotify_artist_id = _get_spotify_artist_id(sp, artist_name)
            time.sleep(SEARCH_DELAY)

        # Scraping Kworb
        kworb_songs = []
        if spotify_artist_id:
            kworb_songs = _scrape_kworb(spotify_artist_id, artist_name)
            time.sleep(SCRAPE_DELAY)
        else:
            logger.warning("    Pas d'ID Spotify → Kworb inaccessible pour %s", artist_name)

        # Matching track par track
        matched = failed = 0
        for track_id, track_name in tracks:
            song = _match_track(track_name, kworb_songs)

            if song:
                con.execute(
                    "INSERT OR REPLACE INTO kworb_streams VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        track_id, artist_name, track_name,
                        song["spotify_track_id"],
                        song["streams"],
                        song["daily"],
                        song["is_feature"],
                        song["title"],
                        song["match_score"],
                        song["last_updated"],
                    ],
                )
                matched += 1
            else:
                # Insère avec streams NULL pour ne pas retraiter
                con.execute(
                    "INSERT OR REPLACE INTO kworb_streams VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                    [track_id, artist_name, track_name],
                )
                failed += 1
                logger.debug("    ❌ Non matché : %s", track_name)

        total_matched += matched
        total_failed  += failed
        logger.info("    ✅ %d matchés | ❌ %d non trouvés sur Kworb", matched, failed)

    con.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Terminé : %d matchés | %d introuvables", total_matched, total_failed)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/kworb_streams.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    Path("logs").mkdir(exist_ok=True)
    run(db_path=DB_PATH)