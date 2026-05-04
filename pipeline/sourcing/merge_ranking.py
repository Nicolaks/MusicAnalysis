"""
merge_ranking.py
================
Fusionne les 4 classements musicaux avec les tracks de la BDD
via fuzzy matching (artiste + titre) et crée une table ranking_data.

Sources :
    - Apple Music EU  : top écoutes Europe depuis 2017
    - iTunes FR       : popularité cumulée France depuis 2013
    - YouTube         : vues totales clips musicaux
    - Spotify FR      : streams cumulés France depuis 2013

Table ranking_data (1 ligne = 1 track présent dans au moins 1 classement) :
    Identité   : track_id, isrc, artist_name, album_name, track_name
    Apple      : apple_days, apple_peak, apple_total, apple_today
    iTunes     : itunes_rank, itunes_popularity
    YouTube    : youtube_views
    Spotify    : spotify_weeks, spotify_top10, spotify_peak,
                 spotify_peak_streams, spotify_total_streams
    Matching   : *_match_str, *_score  (pour audit qualité)
"""

import re
import logging
import duckdb
import pandas as pd

from pathlib import Path
from rapidfuzz import fuzz, process
from unidecode import unidecode

logger = logging.getLogger(__name__)

DB_PATH      = Path("data/warehouse.duckdb")
CSV_DIR      = Path("data/classements")
FUZZY_THRESHOLD = 78

DDL_RANKING = """
CREATE TABLE IF NOT EXISTS ranking_data (
    -- Identité
    track_id               INTEGER PRIMARY KEY,
    isrc                   VARCHAR,
    artist_name            VARCHAR,
    album_name             VARCHAR,
    track_name             VARCHAR,
    release_year           INTEGER,

    -- Apple Music EU
    apple_days             INTEGER,
    apple_peak             INTEGER,
    apple_total            BIGINT,
    apple_today            BIGINT,
    apple_match_str        VARCHAR,
    apple_score            FLOAT,

    -- iTunes FR
    itunes_rank            INTEGER,
    itunes_popularity      FLOAT,
    itunes_match_str       VARCHAR,
    itunes_score           FLOAT,

    -- YouTube
    youtube_views          BIGINT,
    youtube_match_str      VARCHAR,
    youtube_score          FLOAT,

    -- Spotify FR
    spotify_weeks          INTEGER,
    spotify_top10          INTEGER,
    spotify_peak           INTEGER,
    spotify_peak_streams   BIGINT,
    spotify_total_streams  BIGINT,
    spotify_match_str      VARCHAR,
    spotify_score          FLOAT,

    -- Présence dans les classements (pratique pour filtrer)
    in_apple               BOOLEAN DEFAULT FALSE,
    in_itunes              BOOLEAN DEFAULT FALSE,
    in_youtube             BOOLEAN DEFAULT FALSE,
    in_spotify             BOOLEAN DEFAULT FALSE
);
"""


# ─────────────────────────────────────────────
# Nettoyage / normalisation
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = unidecode(str(text)).lower()
    # Retire feat / ft
    text = re.sub(r"\b(feat\.?|ft\.?)\s+[^,\-\)]+", "", text, flags=re.IGNORECASE)
    # Retire parenthèses et crochets
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)
    # Retire ponctuation
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_number(val) -> float | None:
    if pd.isna(val):
        return None
    try:
        return float(re.sub(r"[,\.](?=\d{3})", "", str(val)).replace(",", "."))
    except ValueError:
        return None


def _split(raw: str) -> tuple[str, str]:
    """'Artiste - Titre' → (artiste, titre)"""
    if " - " in raw:
        parts = raw.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return "", raw.strip()


# ─────────────────────────────────────────────
# Chargement CSV
# ─────────────────────────────────────────────

def _load_apple(path: Path) -> pd.DataFrame:
    logger.info("  Chargement Apple Music EU : %s", path.name)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["key"]   = df["Artist and Title"].apply(_normalize)
    df["days"]  = pd.to_numeric(df["Days"],  errors="coerce")
    df["peak"]  = pd.to_numeric(df["Pk"],    errors="coerce")
    df["total"] = df["Total"].apply(_clean_number)
    df["today"] = df["Today"].apply(_clean_number)
    # Rang implicite = position dans le fichier (ligne 0 = plus importante)
    df["implicit_rank"] = df.index + 1
    df[["artist_raw", "title_raw"]] = df["Artist and Title"].apply(
        lambda x: pd.Series(x.split(" - ", 1) if " - " in x else ["", x])
    )
    df["artist_key"] = df["artist_raw"].apply(_normalize)
    df["title_key"]  = df["title_raw"].apply(_normalize)
    logger.info("    %d entrées", len(df))
    return df[["key", "Artist and Title", "artist_key", "title_key", "days", "peak", "total", "today", "implicit_rank"]]


def _load_itunes(path: Path) -> pd.DataFrame:
    logger.info("  Chargement iTunes FR : %s", path.name)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["key"]        = df["Artist and Title"].apply(_normalize)
    df["rank"]       = pd.to_numeric(df["Pos"],        errors="coerce")
    df["popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
    df[["artist_raw", "title_raw"]] = df["Artist and Title"].apply(
        lambda x: pd.Series(x.split(" - ", 1) if " - " in x else ["", x])
    )
    df["artist_key"] = df["artist_raw"].apply(_normalize)
    df["title_key"]  = df["title_raw"].apply(_normalize)
    logger.info("    %d entrées", len(df))
    return df[["key", "Artist and Title", "rank", "popularity", "artist_key", "title_key"]]


def _load_youtube(path: Path) -> pd.DataFrame:
    logger.info("  Chargement YouTube : %s", path.name)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["key"]   = df["Video"].apply(_normalize)
    df["views"] = df["Views"].apply(_clean_number)
    df[["artist_raw", "title_raw"]] = df["Video"].apply(
        lambda x: pd.Series(x.split(" - ", 1) if " - " in x else ["", x])
    )
    df["artist_key"] = df["artist_raw"].apply(_normalize)
    df["title_key"]  = df["title_raw"].apply(_normalize)
    logger.info("    %d entrées", len(df))
    return df[["key", "Video", "views", "artist_key", "title_key"]]


def _load_spotify(path: Path) -> pd.DataFrame:
    logger.info("  Chargement Spotify FR : %s", path.name)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["key"]          = df["Artist and Title"].apply(_normalize)
    df["weeks"]        = pd.to_numeric(df["Wks"], errors="coerce")
    df["top10"]        = pd.to_numeric(df["T10"], errors="coerce")
    df["peak"]         = pd.to_numeric(df["Pk"],  errors="coerce")
    df["peak_streams"] = df["PkStreams"].apply(_clean_number)
    df["total"]        = df["Total"].apply(_clean_number)
    df[["artist_raw", "title_raw"]] = df["Artist and Title"].apply(
        lambda x: pd.Series(x.split(" - ", 1) if " - " in x else ["", x])
    )
    df["artist_key"] = df["artist_raw"].apply(_normalize)
    df["title_key"]  = df["title_raw"].apply(_normalize)
    logger.info("    %d entrées", len(df))
    return df[["key", "Artist and Title", "weeks", "top10", "peak", "peak_streams", "total", "artist_key", "title_key"]]


# ─────────────────────────────────────────────
# Fuzzy matching
# ─────────────────────────────────────────────

def _best_match(
    query_artist: str,
    query_title: str,
    choices_artists: list[str],
    choices_titles: list[str],
    threshold: float = FUZZY_THRESHOLD,
) -> tuple[int | None, float]:
    """
    Match en comparant artiste vs artiste ET titre vs titre séparément.
    query_artist, query_title : déjà normalisés
    choices_artists, choices_titles : listes parallèles, déjà normalisées
    """
    if not query_title or not choices_titles:
        return None, 0.0

    best_idx   = None
    best_score = 0.0

    for i, (csv_artist, csv_title) in enumerate(zip(choices_artists, choices_titles)):
        # Score artiste
        artist_score = fuzz.ratio(query_artist, csv_artist)
        if artist_score < 80:
            continue

        # Score titre
        title_score = fuzz.token_sort_ratio(query_title, csv_title)
        if title_score < 85:
            continue

        # Contrôle longueur titre
        len_q = len(query_title.replace(" ", ""))
        len_c = len(csv_title.replace(" ", ""))
        if len_q == 0 or len_c == 0:
            continue
        if min(len_q, len_c) / max(len_q, len_c) < 0.6:
            continue

        # Score combiné
        combined = artist_score * 0.4 + title_score * 0.6
        if combined > best_score and combined >= threshold:
            best_score = combined
            best_idx   = i

    return best_idx, best_score

def _best_match_spotify(
    query_artist: str,
    query_title: str,
    choices_artists: list[str],
    choices_titles: list[str],
) -> tuple[int | None, float]:
    """
    Matching strict pour Spotify — évite les faux positifs sur titres courts.
    Contraintes supplémentaires :
        - artist_score >= 92  (quasi exact)
        - title_score  >= 92  (quasi exact)
        - longueur titre : ratio >= 0.75
        - titre minimum 3 caractères
    """
    if not query_title or not choices_titles:
        return None, 0.0

    # Titre trop court → risque trop élevé de faux positif
    if len(query_title.replace(" ", "")) < 3:
        return None, 0.0

    best_idx   = None
    best_score = 0.0

    for i, (csv_artist, csv_title) in enumerate(zip(choices_artists, choices_titles)):

        # Artiste quasi exact
        artist_score = fuzz.ratio(query_artist, csv_artist)
        if artist_score < 92:
            continue

        # Titre quasi exact
        title_score = fuzz.token_sort_ratio(query_title, csv_title)
        if title_score < 92:
            continue

        # Longueur titre stricte
        len_q = len(query_title.replace(" ", ""))
        len_c = len(csv_title.replace(" ", ""))
        if len_q == 0 or len_c == 0:
            continue
        if min(len_q, len_c) / max(len_q, len_c) < 0.75:
            continue

        combined = artist_score * 0.4 + title_score * 0.6
        if combined > best_score:
            best_score = combined
            best_idx   = i

    return best_idx, best_score


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

def run(db_path: Path = DB_PATH, csv_dir: Path = CSV_DIR) -> None:
    logger.info("=" * 60)
    logger.info("Démarrage fusion classements")
    logger.info("=" * 60)

    # ── Chargement CSV ────────────────────────
    apple   = _load_apple(csv_dir   / "European_apple_music_song_since_2017.csv")
    itunes  = _load_itunes(csv_dir  / "FR_itunes_cumulatives_popularity_since_2013.csv")
    youtube = _load_youtube(csv_dir / "Most_viwed_music_videos_youtube.csv")
    spotify = _load_spotify(csv_dir / "Spotify_weekly_chart_totals_FR_since_2013.csv")

    apple_artists   = apple["artist_key"].tolist()
    apple_titles    = apple["title_key"].tolist()
    itunes_artists  = itunes["artist_key"].tolist()
    itunes_titles   = itunes["title_key"].tolist()
    youtube_artists = youtube["artist_key"].tolist()
    youtube_titles  = youtube["title_key"].tolist()
    spotify_artists = spotify["artist_key"].tolist()
    spotify_titles  = spotify["title_key"].tolist()

    # ── Chargement BDD ────────────────────────
    con = duckdb.connect(str(db_path))
    con.execute(DDL_RANKING)

    tracks = con.execute("""
        SELECT
            t.track_id,
            i.isrc,
            t.artist_name,
            t.album_name,
            t.track_name,
            t.track_release_year
        FROM tracks_flat t
        LEFT JOIN isrc_data i ON t.track_id = i.track_id
        ORDER BY t.artist_name, t.track_name
    """).fetchall()

    total   = len(tracks)
    matched = {"apple": 0, "itunes": 0, "youtube": 0, "spotify": 0}
    in_any  = 0

    logger.info("%d tracks à matcher contre les classements", total)

    for n, (track_id, isrc, artist_name, album_name, track_name, release_year) in enumerate(tracks, start=1):

        if n % 200 == 0:
            logger.info("  Progression : %d / %d", n, total)

        # Clé de matching : "artiste titre" normalisé
        query_artist = _normalize(artist_name)
        query_title  = _normalize(track_name)

        # ── Apple Music ───────────────────────
        apple_days = apple_peak = apple_total = apple_today = None
        apple_match_str = None
        apple_score = 0.0
        idx, score = _best_match(query_artist, query_title, apple_artists,   apple_titles)
        if idx is not None:
            row = apple.iloc[idx]
            apple_days      = int(row["days"])      if pd.notna(row["days"])  else None
            apple_peak      = int(row["peak"])      if pd.notna(row["peak"])  else None
            apple_total     = int(row["total"])     if pd.notna(row["total"]) else None
            apple_today     = int(row["today"])     if pd.notna(row["today"]) else None
            apple_match_str = row["Artist and Title"]
            apple_score     = score
            matched["apple"] += 1
            logger.info("  [Apple]   %s → %s (%.1f)", query_artist, apple_match_str, apple_score)
        else:
            logger.info("  [Apple]   ❌ %s", query_artist)

        # ── iTunes ────────────────────────────
        itunes_rank = None
        itunes_pop  = None
        itunes_match_str = None
        itunes_score = 0.0
        idx, score = _best_match(query_artist, query_title, itunes_artists,  itunes_titles)
        if idx is not None:
            row = itunes.iloc[idx]
            itunes_rank      = int(row["rank"])       if pd.notna(row["rank"])       else None
            itunes_pop       = float(row["popularity"]) if pd.notna(row["popularity"]) else None
            itunes_match_str = row["Artist and Title"]
            itunes_score     = score
            matched["itunes"] += 1
            logger.info("  [iTunes]  %s → %s (%.1f)", query_artist, itunes_match_str, itunes_score)
        else:
            logger.info("  [iTunes]  ❌ %s", query_artist)

        # ── YouTube ───────────────────────────
        yt_views = None
        youtube_match_str = None
        youtube_score = 0.0
        idx, score = _best_match(query_artist, query_title, youtube_artists, youtube_titles)
        if idx is not None:
            row = youtube.iloc[idx]
            yt_views          = int(row["views"]) if pd.notna(row["views"]) else None
            youtube_match_str = row["Video"]
            youtube_score     = score
            matched["youtube"] += 1
            logger.info("  [YouTube] %s → %s (%.1f)", query_artist, youtube_match_str, youtube_score)
        else:
            logger.info("  [YouTube] ❌ %s", query_artist)

        # ── Spotify ───────────────────────────
        sp_weeks = sp_top10 = sp_peak = sp_peak_streams = sp_total = None
        spotify_match_str = None
        spotify_score = 0.0
        idx, score = _best_match_spotify(query_artist, query_title, spotify_artists, spotify_titles)
        if idx is not None:
            row = spotify.iloc[idx]
            sp_weeks          = int(row["weeks"])        if pd.notna(row["weeks"])        else None
            sp_top10          = int(row["top10"])        if pd.notna(row["top10"])        else None
            sp_peak           = int(row["peak"])         if pd.notna(row["peak"])         else None
            sp_peak_streams   = int(row["peak_streams"]) if pd.notna(row["peak_streams"]) else None
            sp_total          = int(row["total"])        if pd.notna(row["total"])        else None
            spotify_match_str = row["Artist and Title"]
            spotify_score     = score
            matched["spotify"] += 1
            logger.info("  [Spotify] %s → %s (%.1f)", query_artist, spotify_match_str, spotify_score)
        else:
            logger.info("  [Spotify] ❌ %s", query_artist)
        
        # N'insère que les tracks présents dans au moins 1 classement
        in_apple   = apple_match_str   is not None
        in_itunes  = itunes_match_str  is not None
        in_youtube = youtube_match_str is not None
        in_spotify = spotify_match_str is not None

        if not any([in_apple, in_itunes, in_youtube, in_spotify]):
            continue

        in_any += 1

        con.execute(
            "INSERT OR REPLACE INTO ranking_data VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                track_id, isrc, artist_name, album_name, track_name, release_year,
                apple_days, apple_peak, apple_total, apple_today,
                apple_match_str, apple_score,
                itunes_rank, itunes_pop, itunes_match_str, itunes_score,
                yt_views, youtube_match_str, youtube_score,
                sp_weeks, sp_top10, sp_peak, sp_peak_streams, sp_total,
                spotify_match_str, spotify_score,
                in_apple, in_itunes, in_youtube, in_spotify,
            ],
        )

    con.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Terminé : %d / %d tracks présents dans au moins 1 classement", in_any, total)
    logger.info("  Apple Music : %d matchs", matched["apple"])
    logger.info("  iTunes      : %d matchs", matched["itunes"])
    logger.info("  YouTube     : %d matchs", matched["youtube"])
    logger.info("  Spotify     : %d matchs", matched["spotify"])
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/merge_ranking.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    Path("logs").mkdir(exist_ok=True)
    run(db_path=DB_PATH, csv_dir=CSV_DIR)