"""
config.py
=========
Constantes globales : chemins, stopwords, DDL des 3 tables.
"""

from pathlib import Path

import nltk
from nltk.corpus import stopwords

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = Path("data/warehouse.duckdb")

# ─────────────────────────────────────────────────────────────────────────────
# NLTK — téléchargement silencieux si absent
# ─────────────────────────────────────────────────────────────────────────────

def ensure_nltk():
    for res in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


ensure_nltk()

# ─────────────────────────────────────────────────────────────────────────────
# Stopwords
# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS_FR = set(stopwords.words("french"))
STOP_WORDS_EN = set(stopwords.words("english"))

CUSTOM_STOPWORDS = {
    "ouais", "yeah", "hey", "nan", "hein", "han",
    "plus", "tout", "comme", "fait", "faire",
    "bien", "trop", "quand", "toujours",
    "jamais", "fois", "temps", "maintenant",
    "veux", "faut", "dit", "sais", "peu",
    "deux", "rien", "mal", "vie", "monde",
    "fais", "là", "sans", "si", "tous",
    "va", "vais", "ça", "être", "donc",
    "the", "you", "get", "and",
    "sous", "avant", "vois", "ici",
    "oui", "kho", "toutes", "veut",
    "gros", "non", "tiens", "dire",
    "juste", "vrai", "chez", "quoi",
    "leurs", "parle", "parce", "entre",
    "ans", "comment", "après", "depuis",
    "quoi", "car", "dis", "beaucoup", "encore", "sait",
}

ALL_STOPWORDS = STOP_WORDS_FR | CUSTOM_STOPWORDS | STOP_WORDS_EN

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

DDL_TRACK = """
CREATE TABLE IF NOT EXISTS tracks_analysis (

    -- ── Clés ──────────────────────────────────────────────────────────────────
    track_id                INTEGER NOT NULL,
    artist_id               INTEGER NOT NULL,
    album_id                INTEGER,
    track_name              VARCHAR,
    artist_name             VARCHAR,
    album_name              VARCHAR,
    isrc                    VARCHAR,
    artist_isrc             VARCHAR,

    -- ── 2. Statistiques lexicales ─────────────────────────────────────────────
    word_count              INTEGER,
    unique_word_count       INTEGER,
    ttr                     DOUBLE,
    avg_sentence_length     DOUBLE,
    sentence_count          INTEGER,
    top10_words             VARCHAR,    -- JSON

    -- ── 3. Sémantique ─────────────────────────────────────────────────────────
    lda_dominant_topic      INTEGER,
    lda_topic_keywords      VARCHAR,    -- JSON
    tfidf_top_keywords      VARCHAR,    -- JSON
    embedding_norm          DOUBLE,

    -- ── 4. Stylométrie ────────────────────────────────────────────────────────
    pos_noun_ratio          DOUBLE,
    pos_verb_ratio          DOUBLE,
    pos_adj_ratio           DOUBLE,
    pos_adv_ratio           DOUBLE,
    pos_pron_ratio          DOUBLE,
    pronoun_i_ratio         DOUBLE,
    pronoun_we_ratio        DOUBLE,
    pronoun_you_ratio       DOUBLE,

    -- ── 5. Rimes & structure ──────────────────────────────────────────────────
    rhyme_density           DOUBLE,
    avg_syllables_line      DOUBLE,
    repetition_ratio        DOUBLE,
    chorus_detected         BOOLEAN,

    -- ── 6. Lisibilité ─────────────────────────────────────────────────────────
    flesch_reading_ease     DOUBLE,
    flesch_kincaid_grade    DOUBLE,
    smog_index              DOUBLE,
    avg_word_length         DOUBLE,

    -- ── 7. Analyse avancée ────────────────────────────────────────────────────
    semantic_density        DOUBLE,
    lexical_diversity       DOUBLE,
    hapax_ratio             DOUBLE,

    analyzed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (track_id, artist_id)
);
"""

DDL_ALBUM = """
CREATE TABLE IF NOT EXISTS albums_analysis (

    -- ── Clés ──────────────────────────────────────────────────────────────────
    album_id                INTEGER NOT NULL,
    artist_id               INTEGER NOT NULL,
    album_name              VARCHAR,
    artist_name             VARCHAR,
    release_year            INTEGER,
    track_count             INTEGER,

    -- ── 2. Stats lexicales ────────────────────────────────────────────────────
    total_word_count        INTEGER,
    avg_word_count          DOUBLE,
    avg_unique_word_count   DOUBLE,
    avg_ttr                 DOUBLE,
    album_ttr               DOUBLE,
    avg_sentence_length     DOUBLE,
    album_vocabulary_size   INTEGER,
    top20_words             VARCHAR,    -- JSON

    -- ── 3. Sémantique ─────────────────────────────────────────────────────────
    lda_dominant_topic      INTEGER,
    lda_topic_distribution  VARCHAR,    -- JSON {topic_id: prob}
    tfidf_top_keywords      VARCHAR,    -- JSON
    intra_album_similarity  DOUBLE,

    -- ── 4. Stylométrie ────────────────────────────────────────────────────────
    avg_pos_noun_ratio      DOUBLE,
    avg_pos_verb_ratio      DOUBLE,
    avg_pos_adj_ratio       DOUBLE,
    avg_pos_adv_ratio       DOUBLE,
    avg_pos_pron_ratio      DOUBLE,
    avg_pronoun_i_ratio     DOUBLE,
    avg_pronoun_we_ratio    DOUBLE,
    avg_pronoun_you_ratio   DOUBLE,

    -- ── 5. Rimes & structure ──────────────────────────────────────────────────
    avg_rhyme_density       DOUBLE,
    avg_syllables_line      DOUBLE,
    avg_repetition_ratio    DOUBLE,
    pct_with_chorus         DOUBLE,

    -- ── 6. Lisibilité ─────────────────────────────────────────────────────────
    avg_flesch_reading_ease DOUBLE,
    avg_flesch_kincaid_grade DOUBLE,
    avg_smog_index          DOUBLE,
    avg_word_length         DOUBLE,

    -- ── 7. Avancé ─────────────────────────────────────────────────────────────
    avg_semantic_density    DOUBLE,
    avg_lexical_diversity   DOUBLE,
    avg_hapax_ratio         DOUBLE,
    album_hapax_ratio       DOUBLE,

    analyzed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (album_id, artist_id)
);
"""

DDL_ARTIST = """
CREATE TABLE IF NOT EXISTS artists_analysis (

    -- ── Clés ──────────────────────────────────────────────────────────────────
    artist_id               INTEGER PRIMARY KEY,
    artist_name             VARCHAR,
    artist_isrc             VARCHAR,
    album_count             INTEGER,
    track_count             INTEGER,

    -- ── 2. Stats lexicales ────────────────────────────────────────────────────
    total_word_count        INTEGER,
    avg_word_count          DOUBLE,
    career_vocabulary_size  INTEGER,
    career_ttr              DOUBLE,
    avg_ttr                 DOUBLE,
    avg_unique_word_count   DOUBLE,
    avg_sentence_length     DOUBLE,
    top30_words             VARCHAR,    -- JSON

    -- ── 3. Sémantique ─────────────────────────────────────────────────────────
    lda_topic_distribution  VARCHAR,    -- JSON répartition topics carrière
    tfidf_top_keywords      VARCHAR,    -- JSON mots-clés signatures
    inter_album_similarity  DOUBLE,
    career_embedding_centroid VARCHAR,  -- JSON vecteur centroïde PCA 10d

    -- ── 4. Stylométrie ────────────────────────────────────────────────────────
    avg_pos_noun_ratio      DOUBLE,
    avg_pos_verb_ratio      DOUBLE,
    avg_pos_adj_ratio       DOUBLE,
    avg_pos_adv_ratio       DOUBLE,
    avg_pos_pron_ratio      DOUBLE,
    avg_pronoun_i_ratio     DOUBLE,
    avg_pronoun_we_ratio    DOUBLE,
    avg_pronoun_you_ratio   DOUBLE,
    style_signature         VARCHAR,    -- JSON feature vector normalisé

    -- ── 5. Rimes & structure ──────────────────────────────────────────────────
    avg_rhyme_density       DOUBLE,
    std_rhyme_density       DOUBLE,
    avg_syllables_line      DOUBLE,
    avg_repetition_ratio    DOUBLE,
    pct_with_chorus         DOUBLE,

    -- ── 6. Lisibilité ─────────────────────────────────────────────────────────
    avg_flesch_reading_ease DOUBLE,
    avg_flesch_kincaid_grade DOUBLE,
    avg_smog_index          DOUBLE,
    avg_word_length         DOUBLE,

    -- ── 7. Avancé ─────────────────────────────────────────────────────────────
    avg_semantic_density    DOUBLE,
    avg_lexical_diversity   DOUBLE,
    avg_hapax_ratio         DOUBLE,
    career_hapax_ratio      DOUBLE,

    analyzed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

ALL_DDL = [DDL_TRACK, DDL_ALBUM, DDL_ARTIST]
