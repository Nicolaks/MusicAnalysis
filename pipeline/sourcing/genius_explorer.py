import os
import logging
import duckdb
import lyricsgenius
import time
import re

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
DB_PATH = Path("data/warehouse.duckdb")

ARTISTS = ["PLK"]

DDL = """
CREATE TABLE IF NOT EXISTS tracks_flat (
    artist_id          INTEGER,
    artist_name        VARCHAR,
    artist_image_url   VARCHAR,
    artist_followers   BIGINT,
    artist_header_image VARCHAR,
    artist_url         VARCHAR,
    album_id           INTEGER,
    album_name         VARCHAR,
    album_release_date VARCHAR,
    album_release_year INTEGER,
    album_cover_url    VARCHAR,
    album_url          VARCHAR,
    track_id           INTEGER NOT NULL,
    track_name         VARCHAR,
    track_number       INTEGER,
    track_release_date VARCHAR,
    track_release_year INTEGER,
    track_url          VARCHAR,
    track_views        BIGINT,
    featuring          VARCHAR,
    lyrics             VARCHAR,
    is_canonical       BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (track_id, artist_id)
);
"""

def _load_artist_meta(db_path: Path, query: str) -> Optional[dict]:
    """Retourne les métadonnées artiste depuis la BDD, ou None si inconnu."""
    con = duckdb.connect(str(db_path))
    try:
        row = con.execute("""
            SELECT artist_id, artist_name, artist_image_url,
                   artist_followers, artist_header_image, artist_url
            FROM tracks_flat
            WHERE LOWER(artist_name) = LOWER(?)
            LIMIT 1
        """, [query]).fetchone()
    finally:
        con.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "image_url": row[2],
            "followers_count": row[3], "header_image_url": row[4], "url": row[5],
            "albums": [], "loose_tracks": []}

def _get_client() -> lyricsgenius.Genius:
    if not GENIUS_TOKEN:
        raise ValueError("GENIUS_ACCESS_TOKEN manquant dans le .env\n→ https://genius.com/api-clients")
    genius = lyricsgenius.Genius(GENIUS_TOKEN)
    genius.remove_section_headers = True
    genius.retries = 3
    return genius

def _load_existing_tracks(db_path: Path, artist_id: int) -> tuple[dict[int, bool], set[int]]:
    """
    Retourne ({track_id: has_lyrics}, {album_id_complets})
    Un album est "complet" si tous ses tracks ont des paroles.
    """
    con = duckdb.connect(str(db_path))
    
    rows = con.execute("""
        SELECT track_id, lyrics IS NOT NULL
        FROM tracks_flat
        WHERE artist_id = ?
    """, [artist_id]).fetchall()

    # Albums dont TOUS les tracks ont des lyrics
    complete_albums = con.execute("""
        SELECT album_id
        FROM tracks_flat
        WHERE artist_id = ? AND album_id IS NOT NULL
        GROUP BY album_id
        HAVING COUNT(*) > 0 AND SUM(CASE WHEN lyrics IS NULL THEN 1 ELSE 0 END) = 0
    """, [artist_id]).fetchall()
    
    con.close()
    return (
        {tid: has_lyrics for tid, has_lyrics in rows},
        {row[0] for row in complete_albums},
    )

def _parse_date(components: Optional[dict]) -> Optional[str]:
    if not components:
        return None
    y, m, d = components.get("year"), components.get("month"), components.get("day")
    if y and m and d:
        return f"{y:04d}-{m:02d}-{d:02d}"
    if y and m:
        return f"{y:04d}-{m:02d}"
    return str(y) if y else None

def _year(date_str: Optional[str]) -> Optional[int]:
    try:
        return int(date_str[:4]) if date_str else None
    except (ValueError, TypeError):
        return None
    
def _search_song_safe(genius: lyricsgenius.Genius, song_id: int, retries: int = 3) -> Optional[object]:
    for attempt in range(retries):
        try:
            return genius.search_song(song_id=song_id)
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("search_song(%d) échoué (tentative %d/%d) : %s — attente %ds",
                           song_id, attempt + 1, retries, e, wait)
            time.sleep(wait)
    logger.error("search_song(%d) abandonné après %d tentatives", song_id, retries)
    return None
    
def fetch_artist(query: str, genius: lyricsgenius.Genius, db_path: Path) -> dict:
    logger.info("=" * 60)
    logger.info("Fetching artiste : %s", query)
    logger.info("=" * 60)

    artist = _load_artist_meta(db_path, query)
    if artist:
        logger.info("Artiste trouvé en BDD : %s | id=%d", artist["name"], artist["id"])
    else:
        raw = genius.search_artist(query, max_songs=1, get_full_info=True)
        if not raw:
            raise ValueError(f"Aucun artiste trouvé pour : '{query}'")
        artist = {
            "id": raw._body.get("id"), "name": raw.name,
            "image_url": raw.image_url,
            "header_image_url": raw._body.get("header_image_url"),
            "followers_count": raw._body.get("followers_count"),
            "url": raw.url, "albums": [], "loose_tracks": [],
        }
        logger.info("Artiste résolu via API : %s | id=%d", artist["name"], artist["id"])

    existing_tracks, complete_albums = _load_existing_tracks(db_path, artist["id"])
    
    fetched_this_run: set[int] = set()
    
    logger.info("  Récupération des albums...")
    page = 1
    while True:
        albums_raw = genius.artist_albums(artist["id"], per_page=50, page=page)
        batch = albums_raw.get("albums", [])
        if not batch:
            break

        for album_raw in batch:
            release_date = _parse_date(album_raw.get("release_date_components"))
            album = {
                "id": album_raw["id"],
                "name": album_raw["name"],
                "release_date": release_date,
                "release_year": _year(release_date),
                "cover_url": album_raw.get("cover_art_url"),
                "url": album_raw.get("url", ""),
                "tracks": [],
            }
            if album["id"] in complete_albums:
                logger.info("    ⏭ Skip album complet (déjà en BDD) : %s", album["name"])
                artist["albums"].append(album)  # on garde la méta sans re-fetch les tracks
                continue
            logger.info("  [Album] %s | %s", album["name"], album["release_date"] or "?")
            
            skipped_album = 0
            
            tracks_page = 1
            while True:
                tracks_raw  = genius.album_tracks(album["id"], per_page=50, page=tracks_page)
                tracks_batch = tracks_raw.get("tracks", [])
                if not tracks_batch:
                    break

                for item in tracks_batch:
                    song = item.get("song")
                    if not song:
                        continue

                    song_id = song.get("id")

                    if song_id in existing_tracks:
                        if existing_tracks[song_id]:
                            logger.info("    ⏭ Skip (déjà complet) : %s", song.get("title"))
                            skipped_album += 1
                            continue
                        else:
                            logger.info("    🔁 Retry lyrics manquants : %s", song.get("title"))
                            
                    full_song = _search_song_safe(genius, song_id)
                    if full_song:
                        track = _build_track(full_song)
                        album["tracks"].append(track)
                        fetched_this_run.add(song_id)
                        existing_tracks[song_id] = track["lyrics"] is not None

                next_tracks = tracks_raw.get("next_page")
                if not next_tracks:
                    break
                tracks_page = next_tracks

            feats   = sum(1 for t in album["tracks"] if t["featuring"])
            lyrics  = sum(1 for t in album["tracks"] if t["lyrics"])
            logger.info("    └─ %d nouveaux titres | %d featurings | %d paroles", len(album["tracks"]), feats, lyrics)
            artist["albums"].append(album)

        next_page = albums_raw.get("next_page")
        if not next_page:
            break
        page = next_page

    logger.info("  Albums récupérés : %d", len(artist["albums"]))

    logger.info("  Récupération des titres hors-album...")
    songs_page = 1
    while True:
        songs_raw = genius.artist_songs(artist["id"], per_page=50, page=songs_page, sort="popularity")
        batch = songs_raw.get("songs", [])
        if not batch:
            break

        new_on_page = 0  # ← compteur de nouveautés sur cette page
        for song in batch:
            song_id = song.get("id")
            if song_id in fetched_this_run:
                continue
            if song_id in existing_tracks:
                if existing_tracks[song_id]:
                    fetched_this_run.add(song_id)
                    continue
            if song.get("primary_artist", {}).get("id") != artist["id"]:
                continue

            full_song = _search_song_safe(genius, song_id)
            if full_song:
                track = _build_track(full_song)
                artist["loose_tracks"].append(track)
                fetched_this_run.add(song_id)
                existing_tracks[song_id] = track["lyrics"] is not None
                new_on_page += 1

        next_page = songs_raw.get("next_page")
        if not next_page:
            break

        # ← Si toute la page était déjà en BDD, inutile de paginer davantage
        if new_on_page == 0:
            logger.info("    ⏭ Page hors-album entièrement connue, arrêt de la pagination")
            break

        songs_page = next_page
 
    logger.info("  Titres hors-album nouveaux : %d", len(artist["loose_tracks"]))
 
    total  = sum(len(a["tracks"]) for a in artist["albums"]) + len(artist["loose_tracks"])
    lyrics = sum(1 for a in artist["albums"] for t in a["tracks"] if t["lyrics"])
    lyrics += sum(1 for t in artist["loose_tracks"] if t["lyrics"])
    feats  = sum(1 for a in artist["albums"] for t in a["tracks"] if t["featuring"])
    feats  += sum(1 for t in artist["loose_tracks"] if t["featuring"])
 
    logger.info("✅ %s : %d albums | %d nouveaux titres | %d paroles | %d featurings", artist["name"], len(artist["albums"]), total, lyrics, feats)
    return artist

def _build_track(song) -> dict:
    body = song._body
    rd = _parse_date(body.get("release_date_components")) or body.get("release_date")
    return {
        "id":           body.get("id"),
        "name":         song.title,
        "track_number": body.get("track_number"),
        "release_date": rd,
        "release_year": _year(rd),
        "url":          body.get("url", ""),
        "views":        body.get("stats", {}).get("pageviews"),
        "featuring":    [a["name"] for a in (body.get("featured_artists") or [])],
        "lyrics":       song.lyrics,
    }
    
def load_to_duckdb(artist: dict, db_path: Path) -> None:
    logger.info("Loading %s dans DuckDB -> %s", artist["name"], db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(DDL)

    inserted = skipped = 0
    
    NON_CANONICAL_PATTERNS = re.compile(
        r'[\(\[（]\s*'
        r'(remix|live|freestyle|version|demo|edit|solo|acoustic|instrumental|reprise)\b'
        r'[^)\]）]*'
        r'[\)\]）]',
        re.IGNORECASE
    )
    def _is_canonical(track_name: str) -> bool:
        return not bool(NON_CANONICAL_PATTERNS.search(track_name))

    def insert(track: dict, album: Optional[dict]) -> None:
        nonlocal inserted, skipped
        row = [
            artist["id"], artist["name"], artist["image_url"],
            artist.get("followers_count"),
            artist.get("header_image_url"),
            artist["url"],
            album["id"]           if album else None,
            album["name"]         if album else None,
            album["release_date"] if album else None,
            album["release_year"] if album else None,
            album["cover_url"]    if album else None,
            album["url"]          if album else None,
            track["id"], track["name"], track["track_number"],
            track["release_date"], track["release_year"],
            track["url"], track["views"],
            ", ".join(track["featuring"]) if track["featuring"] else None,
            track["lyrics"],
            _is_canonical(track["name"]),
        ]
        try:
            con.execute(
                "INSERT OR REPLACE INTO tracks_flat VALUES (%s)" % ",".join(["?"] * len(row)),
                row,
            )
            inserted += 1
        except Exception as e:
            logger.error("  Erreur INSERT [%s] : %s", track["name"], e)
            skipped += 1

    for album in artist["albums"]:
        for track in album["tracks"]:
            insert(track, album)
    for track in artist["loose_tracks"]:
        insert(track, None)

    con.close()
    logger.info("✅ DuckDB : %d insérées | %d ignorées | %s", inserted, skipped, artist["name"])
    
def run(artists: list[str], db_path: Path = DB_PATH) -> None:
    logger.info("Démarrage — %d artiste(s) : %s", len(artists), artists)
    genius          = _get_client()
    success, errors = [], []

    for i, query in enumerate(artists, start=1):
        logger.info("")
        logger.info(">>> Artiste %d / %d : %s", i, len(artists), query)
        try:
            artist = fetch_artist(query, genius, db_path)
            load_to_duckdb(artist, db_path)
            success.append(artist["name"])
        except Exception as e:
            logger.error("❌ Échec pour '%s' : %s", query, e, exc_info=True)
            errors.append(query)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Terminé : %d succès | %d erreurs", len(success), len(errors))
    if success:
        logger.info("  ✅ %s", ", ".join(success))
    if errors:
        logger.warning("  ❌ %s", ", ".join(errors))
    logger.info("  Base : %s", db_path.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler("logs/genius_explorer.log", encoding="utf-8"), logging.StreamHandler()])
    run(artists=sys.argv[1:] if len(sys.argv) > 1 else ARTISTS, db_path=DB_PATH)