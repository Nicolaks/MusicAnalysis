"""
audio_features.py
=================
Extrait les features audio depuis les extraits MP3 de 30s (data/samples/)
et les stocke dans une table audio_features_local.

Features extraites via librosa :
    Rythme      : tempo (BPM), beat_strength, rhythm_regularity
    Énergie     : rms_mean, rms_std, dynamic_range
    Spectre     : spectral_centroid, spectral_bandwidth,
                  spectral_rolloff, spectral_flatness, zcr_mean
    Tonalité    : key, mode (majeur/mineur), key_confidence, chroma_std
    Timbre      : mfcc_1..13 (13 coefficients cepstraux)
    Structure   : onset_rate (densité des attaques)
    Ambiance    : brightness, warmth, roughness
"""

import logging
import warnings
import duckdb
import numpy as np

from pathlib import Path
from typing  import Optional

warnings.filterwarnings("ignore")   # librosa est verbeux sur les MP3

logger = logging.getLogger(__name__)

DB_PATH     = Path("data/warehouse.duckdb")
SAMPLES_DIR = Path("data/samples")

DDL_AUDIO = """
CREATE TABLE IF NOT EXISTS audio_features_local (
    track_id            INTEGER PRIMARY KEY,
    artist_name         VARCHAR,
    track_name          VARCHAR,
    file_path           VARCHAR,
    duration_s          FLOAT,

    -- Rythme
    tempo               FLOAT,    -- BPM estimé
    beat_strength       FLOAT,    -- Force moyenne des beats (0-1)
    rhythm_regularity   FLOAT,    -- Régularité du rythme (std des inter-beats, bas = régulier)

    -- Énergie
    rms_mean            FLOAT,    -- Énergie RMS moyenne
    rms_std             FLOAT,    -- Variabilité de l'énergie
    dynamic_range       FLOAT,    -- max RMS - min RMS (plage dynamique)

    -- Spectre fréquentiel
    spectral_centroid   FLOAT,    -- Centre de gravité spectral (Hz) — brillance
    spectral_bandwidth  FLOAT,    -- Largeur du spectre (Hz)
    spectral_rolloff    FLOAT,    -- Fréquence de rolloff 85% (Hz)
    spectral_flatness   FLOAT,    -- Platitude (0=tonal, 1=bruit)
    zcr_mean            FLOAT,    -- Taux de passage par zéro (percussivité)

    -- Tonalité
    key                 INTEGER,  -- 0=Do, 1=Do#, ..., 11=Si
    key_name            VARCHAR,  -- Nom de la tonalité (ex: "La mineur")
    mode                INTEGER,  -- 1=majeur, 0=mineur
    key_confidence      FLOAT,    -- Confiance dans l'estimation (0-1)
    chroma_std          FLOAT,    -- Variabilité chromatique (complexité harmonique)

    -- Timbre (MFCC 1-13)
    mfcc_1              FLOAT,
    mfcc_2              FLOAT,
    mfcc_3              FLOAT,
    mfcc_4              FLOAT,
    mfcc_5              FLOAT,
    mfcc_6              FLOAT,
    mfcc_7              FLOAT,
    mfcc_8              FLOAT,
    mfcc_9              FLOAT,
    mfcc_10             FLOAT,
    mfcc_11             FLOAT,
    mfcc_12             FLOAT,
    mfcc_13             FLOAT,

    -- Structure
    onset_rate          FLOAT,    -- Nb d'attaques par seconde (densité rythmique)

    -- Ambiance dérivée
    brightness          FLOAT,    -- Ratio énergie hautes freq / total (0-1)
    warmth              FLOAT,    -- Ratio énergie basses freq / total (0-1)
    roughness           FLOAT     -- Irrégularité spectrale (distorsion, saturation)
);
"""

KEY_NAMES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def _extract_features(file_path: str) -> Optional[dict]:
    """
    Extrait toutes les features audio d'un fichier MP3.
    Retourne un dict ou None en cas d'erreur.
    """
    try:
        import librosa
    except ImportError:
        raise SystemExit("❌ librosa manquant — pip install librosa")

    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
    except Exception as e:
        logger.error("    Chargement échoué [%s] : %s", file_path, e)
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    feat     = {"duration_s": round(float(duration), 2)}

    # ── Rythme ────────────────────────────────
    tempo, beats      = librosa.beat.beat_track(y=y, sr=sr)
    feat["tempo"] = round(float(np.atleast_1d(tempo)[0]), 2)

    if len(beats) > 1:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strengths = onset_env[beats[beats < len(onset_env)]]
        feat["beat_strength"] = round(float(np.mean(beat_strengths)), 4) if len(beat_strengths) else 0.0
        inter_beats = np.diff(beats)
        feat["rhythm_regularity"] = round(float(np.std(inter_beats)), 4)
    else:
        feat["beat_strength"]     = 0.0
        feat["rhythm_regularity"] = 0.0

    # ── Énergie ───────────────────────────────
    rms               = librosa.feature.rms(y=y)[0]
    feat["rms_mean"]  = round(float(np.mean(rms)), 6)
    feat["rms_std"]   = round(float(np.std(rms)),  6)
    feat["dynamic_range"] = round(float(np.max(rms) - np.min(rms)), 6)

    # ── Spectre ───────────────────────────────
    stft = np.abs(librosa.stft(y))

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    feat["spectral_centroid"] = round(float(np.mean(centroid)), 2)

    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    feat["spectral_bandwidth"] = round(float(np.mean(bandwidth)), 2)

    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    feat["spectral_rolloff"] = round(float(np.mean(rolloff)), 2)

    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    feat["spectral_flatness"] = round(float(np.mean(flatness)), 6)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feat["zcr_mean"] = round(float(np.mean(zcr)), 6)

    # ── Tonalité ──────────────────────────────
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Corrélation avec profils majeur/mineur (Krumhansl)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_scores = [np.corrcoef(np.roll(chroma_mean, -i), major_profile)[0, 1]
                    for i in range(12)]
    minor_scores = [np.corrcoef(np.roll(chroma_mean, -i), minor_profile)[0, 1]
                    for i in range(12)]

    best_major_key   = int(np.argmax(major_scores))
    best_minor_key   = int(np.argmax(minor_scores))
    best_major_score = float(np.max(major_scores))
    best_minor_score = float(np.max(minor_scores))

    if best_major_score >= best_minor_score:
        feat["key"]            = best_major_key
        feat["mode"]           = 1
        feat["key_confidence"] = round(best_major_score, 4)
        feat["key_name"]       = f"{KEY_NAMES[best_major_key]} majeur"
    else:
        feat["key"]            = best_minor_key
        feat["mode"]           = 0
        feat["key_confidence"] = round(best_minor_score, 4)
        feat["key_name"]       = f"{KEY_NAMES[best_minor_key]} mineur"

    feat["chroma_std"] = round(float(np.std(chroma_mean)), 4)

    # ── MFCC (timbre) ─────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for j in range(13):
        feat[f"mfcc_{j+1}"] = round(float(np.mean(mfccs[j])), 4)

    # ── Structure ─────────────────────────────
    onsets             = librosa.onset.onset_detect(y=y, sr=sr)
    feat["onset_rate"] = round(float(len(onsets) / duration), 4) if duration > 0 else 0.0

    # ── Ambiance dérivée ──────────────────────
    freqs      = librosa.fft_frequencies(sr=sr)
    mag        = np.mean(np.abs(librosa.stft(y)), axis=1)
    total_e    = np.sum(mag) + 1e-10

    bright_mask          = freqs > 4000
    feat["brightness"]   = round(float(np.sum(mag[bright_mask]) / total_e), 4)

    warm_mask            = freqs < 500
    feat["warmth"]       = round(float(np.sum(mag[warm_mask]) / total_e), 4)

    # Roughness : irrégularité spectrale (flux entre frames adjacentes)
    spec_flux            = np.mean(np.abs(np.diff(stft, axis=1)))
    feat["roughness"]    = round(float(spec_flux), 6)

    return feat


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

def run(db_path: Path = DB_PATH) -> None:
    logger.info("=" * 60)
    logger.info("Démarrage extraction features audio")
    logger.info("=" * 60)

    con = duckdb.connect(str(db_path))
    con.execute(DDL_AUDIO)

    already = con.execute(
        "SELECT COUNT(*) FROM audio_features_local"
    ).fetchone()[0]
    logger.info("Cache audio_features_local : %d tracks déjà traités", already)

    # Tracks avec sample téléchargé, pas encore analysés
    rows = con.execute("""
        SELECT s.track_id, s.artist_name, s.track_name, s.file_path
        FROM samples_index s
        WHERE s.downloaded = 'YES'
          AND s.file_path IS NOT NULL
          AND s.track_id NOT IN (SELECT track_id FROM audio_features_local)
        ORDER BY s.artist_name, s.track_name
    """).fetchall()

    total     = len(rows)
    processed = 0
    failed    = 0

    logger.info("%d sample(s) à analyser", total)

    for i, (track_id, artist_name, track_name, file_path) in enumerate(rows, start=1):
        logger.info("[%d/%d] %s — %s", i, total, artist_name, track_name)

        if not Path(file_path).exists():
            logger.warning("    Fichier introuvable : %s", file_path)
            failed += 1
            continue

        feat = _extract_features(file_path)

        if not feat:
            failed += 1
            continue

        con.execute(
            "INSERT OR REPLACE INTO audio_features_local VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                track_id, artist_name, track_name, file_path,
                feat["duration_s"],
                # Rythme
                feat["tempo"], feat["beat_strength"], feat["rhythm_regularity"],
                # Énergie
                feat["rms_mean"], feat["rms_std"], feat["dynamic_range"],
                # Spectre
                feat["spectral_centroid"], feat["spectral_bandwidth"],
                feat["spectral_rolloff"], feat["spectral_flatness"], feat["zcr_mean"],
                # Tonalité
                feat["key"], feat["key_name"], feat["mode"], feat["key_confidence"],
                feat["chroma_std"],
                # MFCC
                feat["mfcc_1"],  feat["mfcc_2"],  feat["mfcc_3"],  feat["mfcc_4"],
                feat["mfcc_5"],  feat["mfcc_6"],  feat["mfcc_7"],  feat["mfcc_8"],
                feat["mfcc_9"],  feat["mfcc_10"], feat["mfcc_11"], feat["mfcc_12"],
                feat["mfcc_13"],
                # Structure
                feat["onset_rate"],
                # Ambiance
                feat["brightness"], feat["warmth"], feat["roughness"],
            ],
        )
        processed += 1
        logger.info(
            "    ✅ BPM=%.0f | key=%s | energy=%.4f | brightness=%.3f",
            feat["tempo"], feat["key_name"], feat["rms_mean"], feat["brightness"],
        )

    con.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Terminé : %d analysés | %d erreurs", processed, failed)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/audio_features.log", encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    Path("logs").mkdir(exist_ok=True)
    run(db_path=DB_PATH)