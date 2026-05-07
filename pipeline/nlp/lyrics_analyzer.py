"""
lyrics_analyzer.py
==================
Pipeline NLP des paroles — 3 niveaux d'analyse :

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
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import duckdb
import nltk
import numpy as np
import spacy
import textstat
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pronouncing
import pyphen

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = Path("data/warehouse.duckdb")

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
# DDL — 3 tables
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
    album_ttr               DOUBLE,     -- TTR sur texte concaténé album
    avg_sentence_length     DOUBLE,
    album_vocabulary_size   INTEGER,
    top20_words             VARCHAR,    -- JSON

    -- ── 3. Sémantique ─────────────────────────────────────────────────────────
    lda_dominant_topic      INTEGER,
    lda_topic_distribution  VARCHAR,    -- JSON {topic_id: prob}
    tfidf_top_keywords      VARCHAR,    -- JSON
    intra_album_similarity  DOUBLE,     -- cohérence thématique (cosine moyen)

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
    inter_album_similarity  DOUBLE,     -- cohérence stylistique inter-albums
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

# ─────────────────────────────────────────────────────────────────────────────
# NLTK
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_nltk():
    for res in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
            
STOP_WORDS_FR = set(stopwords.words("french"))

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
    "juste", "vrai",
}
STOP_WORDS_EN = set(stopwords.words("english"))

ALL_STOPWORDS = STOP_WORDS_FR.union(CUSTOM_STOPWORDS).union(STOP_WORDS_EN)

def custom_tokenizer(text):
    return _filtered_tokens(text)

def _filtered_tokens(text):
    if isinstance(text, list):
        tokens = [t.lower() for t in text]
    else:
        tokens = word_tokenize(text.lower(), language="french")

    return [
        t for t in tokens
        if (
            t.isalpha()
            and t not in ALL_STOPWORDS
            and len(t) > 2
        )
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_dic = pyphen.Pyphen(lang="fr_FR")

def _count_syllables(word: str) -> int:
    h = _dic.inserted(word)
    if "-" in h:
        return h.count("-") + 1
    return max(1, len(re.findall(r"[aeiouàâäéèêëîïôöùûü]", word.lower())))

def _clean_lyrics(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def _yule_k(tokens: list[str]) -> float:
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    m1 = len(tokens)
    m2 = sum(v ** 2 for v in freq.values())
    return _safe_div(10_000 * (m2 - m1), m1 ** 2) if m1 else 0.0

def _shannon_entropy(values: list[float]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * np.log2(p) for p in probs)

def _linear_trend(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    xs_a, ys_a = np.array(xs, dtype=float), np.array(ys, dtype=float)
    xs_c = xs_a - xs_a.mean()
    denom = (xs_c ** 2).sum()
    return float((xs_c * ys_a).sum() / denom) if denom else 0.0

def _avg(lst: list, default=0.0):
    return float(np.mean(lst)) if lst else default

def _std(lst: list, default=0.0):
    return float(np.std(lst)) if lst else default

# ─────────────────────────────────────────────────────────────────────────────
# Modèles (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_nlp = _sbert_inst = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("fr_core_news_sm absent — fallback en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def _get_sbert():
    global _sbert_inst
    if _sbert_inst is None:
        _sbert_inst = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _sbert_inst

# ─────────────────────────────────────────────────────────────────────────────
# LDA
# ─────────────────────────────────────────────────────────────────────────────

class LDAModel:
    """Wrapper sklearn LDA — même interface que les appels gensim dans le pipeline."""

    def __init__(self, num_topics: int = 12):
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(
            tokenizer=custom_tokenizer,
            min_df=3,
            max_df=0.70,
            lowercase=True,
        )
        self.lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=50,
            learning_method="batch",
            random_state=42,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )
        self.vocab: list[str] = []

    def fit(self, corpus_texts: list[str]) -> "LDAModel":
        """Entraîne sur une liste de textes bruts."""
        dtm = self.vectorizer.fit_transform(corpus_texts)
        self.lda.fit(dtm)
        self.vocab = self.vectorizer.get_feature_names_out().tolist()
        return self

    def transform(self, text: str) -> list[tuple[int, float]]:
        """Retourne [(topic_id, prob), …] pour un texte."""
        vec = self.vectorizer.transform([text])
        probs = self.lda.transform(vec)[0]
        return [(i, float(p)) for i, p in enumerate(probs)]

    def top_words(self, topic_id: int, topn: int = 5) -> list[str]:
        """Mots les plus probables d'un topic."""
        comp = self.lda.components_[topic_id]
        indices = comp.argsort()[-topn:][::-1]
        return [self.vocab[i] for i in indices]


def build_lda(corpus_texts: list[str], num_topics: int = 8) -> LDAModel:
    """Entraîne et retourne un LDAModel sklearn sur le corpus."""
    model = LDAModel(num_topics=num_topics)
    model.fit(corpus_texts)
    logger.info("LDA entraîné — %d topics | vocabulaire %d mots", num_topics, len(model.vocab))
    return model

# ─────────────────────────────────────────────────────────────────────────────
# NIVEAU 1 — Analyse d'une chanson
# ─────────────────────────────────────────────────────────────────────────────

def analyze_track(
    track_id: int, artist_id: int, album_id: Optional[int],
    track_name: str, artist_name: str, album_name: Optional[str],
    raw_lyrics: str,
    isrc: Optional[str] = None,
    artist_isrc: Optional[str] = None,
    lda_model=None,
) -> dict:

    lyrics = _clean_lyrics(raw_lyrics)
    r: dict = {
        "track_id": track_id, "artist_id": artist_id, "album_id": album_id,
        "track_name": track_name, "artist_name": artist_name, "album_name": album_name,
        "isrc": isrc, "artist_isrc": artist_isrc,
    }

    # 2. Stats lexicales
    tokens = word_tokenize(lyrics.lower(), language="french")
    alpha  = _filtered_tokens(lyrics)
    sents  = sent_tokenize(lyrics, language="french")
    r["word_count"]          = len(alpha)
    r["unique_word_count"]   = len(set(alpha))
    r["ttr"]                 = _safe_div(len(set(alpha)), len(alpha))
    r["sentence_count"]      = len(sents)
    r["avg_sentence_length"] = _safe_div(len(alpha), len(sents))
    freq = FreqDist(alpha)
    r["top10_words"] = json.dumps([w for w, _ in freq.most_common(10)], ensure_ascii=False)

    # 3. Sémantique
    try:
        tfidf = TfidfVectorizer(max_features=10, stop_words=list(ALL_STOPWORDS), tokenizer=custom_tokenizer)
        tfidf.fit_transform([lyrics])
        r["tfidf_top_keywords"] = json.dumps(tfidf.get_feature_names_out().tolist(), ensure_ascii=False)
    except Exception:
        r["tfidf_top_keywords"] = None

    if lda_model:
        try:
            topics = lda_model.transform(lyrics)
            best   = max(topics, key=lambda x: x[1])
            r["lda_dominant_topic"] = int(best[0])
            r["lda_topic_keywords"] = json.dumps(
                lda_model.top_words(best[0], topn=5), ensure_ascii=False
            )
        except Exception:
            r["lda_dominant_topic"] = r["lda_topic_keywords"] = None
    else:
        r["lda_dominant_topic"] = r["lda_topic_keywords"] = None

    try:
        emb = _get_sbert().encode(lyrics[:1000])
        r["embedding_norm"] = float(np.linalg.norm(emb))
    except Exception:
        r["embedding_norm"] = None

    # 4. Stylométrie spaCy
    doc   = _get_nlp()(lyrics[:100_000])
    total = sum(1 for t in doc if not t.is_space)
    pos_c: dict[str, int] = {}
    for t in doc:
        if not t.is_space:
            pos_c[t.pos_] = pos_c.get(t.pos_, 0) + 1
    for pos, key in [("NOUN","noun"),("VERB","verb"),("ADJ","adj"),("ADV","adv"),("PRON","pron")]:
        r[f"pos_{key}_ratio"] = _safe_div(pos_c.get(pos, 0), total)
    r["pronoun_i_ratio"]   = _safe_div(sum(1 for t in doc if t.lower_ in {"je","j'"}), total)
    r["pronoun_we_ratio"]  = _safe_div(sum(1 for t in doc if t.lower_ in {"on","nous"}), total)
    r["pronoun_you_ratio"] = _safe_div(sum(1 for t in doc if t.lower_ in {"tu","vous","t'"}), total)

    # 5. Rimes & structure
    lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    rhyme_count = 0
    syl_per_line = []
    for line in lines:
        ws = line.split()
        if ws:
            last = re.sub(r"[^\w]", "", ws[-1]).lower()
            syl_per_line.append(sum(_count_syllables(w) for w in ws))
            if len(last) >= 3:
                rhyme_count += sum(
                    1 for other in lines
                    if other != line and other.split() and
                    re.sub(r"[^\w]", "", other.split()[-1]).lower()[-3:] == last[-3:]
                )
    r["rhyme_density"]      = _safe_div(rhyme_count, len(lines))
    r["avg_syllables_line"] = _avg(syl_per_line)
    line_counts = Counter(lines)
    repeated = sum(v for v in line_counts.values() if v >= 3)
    r["repetition_ratio"] = _safe_div(repeated, len(lines))
    r["chorus_detected"]  = repeated > 0

    # 6. Lisibilité
    r["flesch_reading_ease"]  = textstat.flesch_reading_ease(lyrics)
    r["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(lyrics)
    r["smog_index"]           = textstat.smog_index(lyrics)
    r["avg_word_length"]      = _safe_div(sum(len(t) for t in alpha), len(alpha))

    # 7. Avancé
    content = [t for t in doc if t.pos_ in {"NOUN","VERB","ADJ","ADV"} and not t.is_space]
    r["semantic_density"]  = _safe_div(len(content), total)
    r["lexical_diversity"] = _yule_k(alpha)
    fd2 = FreqDist(alpha)
    hapax = sum(1 for _, c in fd2.items() if c == 1)
    r["hapax_ratio"] = _safe_div(hapax, len(set(alpha)))

    return r

# ─────────────────────────────────────────────────────────────────────────────
# NIVEAU 2 — Agrégation album
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_album(
    album_id: int, artist_id: int,
    album_name: str, artist_name: str,
    release_year: Optional[int],
    track_rows: list[dict],
    raw_lyrics_list: list[str],
    lda_model=None,
) -> dict:

    n = len(track_rows)
    r: dict = {
        "album_id": album_id, "artist_id": artist_id,
        "album_name": album_name, "artist_name": artist_name,
        "release_year": release_year, "track_count": n,
    }
    if n == 0:
        return r

    def _col(key): return [t[key] for t in track_rows if t.get(key) is not None]

    # 2. Stats lexicales
    r["total_word_count"]      = int(sum(_col("word_count")))
    r["avg_word_count"]        = _avg(_col("word_count"))
    r["avg_unique_word_count"] = _avg(_col("unique_word_count"))
    r["avg_ttr"]               = _avg(_col("ttr"))
    r["avg_sentence_length"]   = _avg(_col("avg_sentence_length"))
    combined = " ".join(_clean_lyrics(l) for l in raw_lyrics_list)
    all_tokens = _filtered_tokens(combined)
    r["album_vocabulary_size"] = len(set(all_tokens))
    r["album_ttr"]             = _safe_div(len(set(all_tokens)), len(all_tokens))
    freq_all = FreqDist(all_tokens)
    r["top20_words"] = json.dumps([w for w, _ in freq_all.most_common(20)], ensure_ascii=False)

    # 3. Sémantique
    try:
        tfidf = TfidfVectorizer(max_features=15, stop_words=list(ALL_STOPWORDS))
        tfidf.fit_transform([combined])
        r["tfidf_top_keywords"] = json.dumps(tfidf.get_feature_names_out().tolist(), ensure_ascii=False)
    except Exception:
        r["tfidf_top_keywords"] = None

    if lda_model:
        try:
            topics = lda_model.transform(combined)
            best   = max(topics, key=lambda x: x[1])
            r["lda_dominant_topic"]     = int(best[0])
            r["lda_topic_distribution"] = json.dumps({str(tid): float(p) for tid, p in topics})
        except Exception:
            r["lda_dominant_topic"] = r["lda_topic_distribution"] = None
    else:
        r["lda_dominant_topic"] = r["lda_topic_distribution"] = None

    try:
        sbert = _get_sbert()
        embs  = sbert.encode([_clean_lyrics(l)[:512] for l in raw_lyrics_list])
        if len(embs) > 1:
            sim_mat = cosine_similarity(embs)
            mask = np.ones(sim_mat.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            r["intra_album_similarity"] = float(sim_mat[mask].mean())
        else:
            r["intra_album_similarity"] = 1.0
    except Exception:
        r["intra_album_similarity"] = None

    # 4–7. Agrégations
    for key in ["pos_noun_ratio","pos_verb_ratio","pos_adj_ratio","pos_adv_ratio","pos_pron_ratio",
                "pronoun_i_ratio","pronoun_we_ratio","pronoun_you_ratio"]:
        r[f"avg_{key}"] = _avg(_col(key))

    r["avg_rhyme_density"]    = _avg(_col("rhyme_density"))
    r["avg_syllables_line"]   = _avg(_col("avg_syllables_line"))
    r["avg_repetition_ratio"] = _avg(_col("repetition_ratio"))
    chorus_vals = [t["chorus_detected"] for t in track_rows if t.get("chorus_detected") is not None]
    r["pct_with_chorus"]      = _safe_div(sum(chorus_vals), len(chorus_vals)) if chorus_vals else 0.0

    r["avg_flesch_reading_ease"]  = _avg(_col("flesch_reading_ease"))
    r["avg_flesch_kincaid_grade"] = _avg(_col("flesch_kincaid_grade"))
    r["avg_smog_index"]           = _avg(_col("smog_index"))
    r["avg_word_length"]          = _avg(_col("avg_word_length"))

    r["avg_semantic_density"]  = _avg(_col("semantic_density"))
    r["avg_lexical_diversity"] = _avg(_col("lexical_diversity"))
    r["avg_hapax_ratio"]       = _avg(_col("hapax_ratio"))
    hapax = sum(1 for _, c in freq_all.items() if c == 1)
    r["album_hapax_ratio"]     = _safe_div(hapax, len(set(all_tokens)))

    return r

# ─────────────────────────────────────────────────────────────────────────────
# NIVEAU 3 — Agrégation artiste
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_artist(
    artist_id: int, artist_name: str, artist_isrc: Optional[str],
    track_rows: list[dict],
    album_rows: list[dict],
    raw_lyrics_list: list[str],
    lda_model=None,
) -> dict:

    n_tracks = len(track_rows)
    n_albums = len(album_rows)
    r: dict = {
        "artist_id": artist_id, "artist_name": artist_name,
        "artist_isrc": artist_isrc,
        "album_count": n_albums, "track_count": n_tracks,
    }
    if n_tracks == 0:
        return r

    def _col(key): return [t[key] for t in track_rows if t.get(key) is not None]

    # 2. Stats lexicales
    r["total_word_count"]      = int(sum(_col("word_count")))
    r["avg_word_count"]        = _avg(_col("word_count"))
    r["avg_unique_word_count"] = _avg(_col("unique_word_count"))
    r["avg_ttr"]               = _avg(_col("ttr"))
    r["avg_sentence_length"]   = _avg(_col("avg_sentence_length"))
    combined = " ".join(_clean_lyrics(l) for l in raw_lyrics_list)
    all_tokens = _filtered_tokens(combined)
    r["career_vocabulary_size"] = len(set(all_tokens))
    r["career_ttr"]             = _safe_div(len(set(all_tokens)), len(all_tokens))
    clean_tokens = _filtered_tokens(combined)
    freq_all = FreqDist(clean_tokens)
    r["top30_words"] = json.dumps([w for w, _ in freq_all.most_common(30)], ensure_ascii=False)

    # 3. Sémantique
    try:
        tfidf = TfidfVectorizer(max_features=20, stop_words=list(ALL_STOPWORDS))
        tfidf.fit_transform([combined])
        r["tfidf_top_keywords"] = json.dumps(tfidf.get_feature_names_out().tolist(), ensure_ascii=False)
    except Exception:
        r["tfidf_top_keywords"] = None

    if lda_model:
        try:
            topics = lda_model.transform(combined)
            r["lda_topic_distribution"] = json.dumps({str(tid): float(p) for tid, p in topics})
        except Exception:
            r["lda_topic_distribution"] = None
    else:
        r["lda_topic_distribution"] = None

    # Cohérence inter-albums + centroïde embedding
    try:
        sbert = _get_sbert()
        all_embs = sbert.encode([_clean_lyrics(l)[:512] for l in raw_lyrics_list])

        # Inter-album similarity : embedding moyen par album
        chunk = max(1, len(raw_lyrics_list) // max(n_albums, 1))
        album_embs = np.array([
            all_embs[i*chunk:(i+1)*chunk].mean(axis=0)
            for i in range(n_albums)
            if i*chunk < len(all_embs)
        ])
        if len(album_embs) > 1:
            sim_mat = cosine_similarity(album_embs)
            mask = np.ones(sim_mat.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            r["inter_album_similarity"] = float(sim_mat[mask].mean())
        else:
            r["inter_album_similarity"] = 1.0

        # Centroïde réduit PCA 10d
        centroid = all_embs.mean(axis=0)
        if len(all_embs) >= 10:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10)
            pca.fit(all_embs)
            centroid_reduced = pca.transform(centroid.reshape(1, -1))[0]
        else:
            centroid_reduced = centroid[:10]
        r["career_embedding_centroid"] = json.dumps(centroid_reduced.tolist())
    except Exception as e:
        logger.warning("Embedding artiste échoué : %s", e)
        r["inter_album_similarity"] = r["career_embedding_centroid"] = None

    # 4. Stylométrie
    for key in ["pos_noun_ratio","pos_verb_ratio","pos_adj_ratio","pos_adv_ratio","pos_pron_ratio",
                "pronoun_i_ratio","pronoun_we_ratio","pronoun_you_ratio"]:
        r[f"avg_{key}"] = _avg(_col(key))

    style_keys = ["pos_noun_ratio","pos_verb_ratio","pos_adj_ratio","pos_pron_ratio",
                  "pronoun_i_ratio","pronoun_we_ratio","pronoun_you_ratio",
                  "rhyme_density","avg_syllables_line","ttr","semantic_density"]
    r["style_signature"] = json.dumps({k: _avg(_col(k)) for k in style_keys})

    # 5. Rimes
    r["avg_rhyme_density"]    = _avg(_col("rhyme_density"))
    r["std_rhyme_density"]    = _std(_col("rhyme_density"))
    r["avg_syllables_line"]   = _avg(_col("avg_syllables_line"))
    r["avg_repetition_ratio"] = _avg(_col("repetition_ratio"))
    chorus_vals = [t["chorus_detected"] for t in track_rows if t.get("chorus_detected") is not None]
    r["pct_with_chorus"]      = _safe_div(sum(chorus_vals), len(chorus_vals)) if chorus_vals else 0.0

    # 6. Lisibilité
    r["avg_flesch_reading_ease"]  = _avg(_col("flesch_reading_ease"))
    r["avg_flesch_kincaid_grade"] = _avg(_col("flesch_kincaid_grade"))
    r["avg_smog_index"]           = _avg(_col("smog_index"))
    r["avg_word_length"]          = _avg(_col("avg_word_length"))

    # 7. Avancé
    r["avg_semantic_density"]  = _avg(_col("semantic_density"))
    r["avg_lexical_diversity"] = _avg(_col("lexical_diversity"))
    r["avg_hapax_ratio"]       = _avg(_col("hapax_ratio"))
    hapax = sum(1 for _, c in freq_all.items() if c == 1)
    r["career_hapax_ratio"]    = _safe_div(hapax, len(set(all_tokens)))

    return r

# ─────────────────────────────────────────────────────────────────────────────
# DuckDB
# ─────────────────────────────────────────────────────────────────────────────

def _init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    for ddl in ALL_DDL:
        con.execute(ddl)
    return con

def load_tracks_to_analyze(db_path: Path, artists: list[str], rerun: bool) -> list[dict]:
    con = _init_db(db_path)
    artist_filter = ""
    params: list = []
    if artists:
        artist_filter = "AND LOWER(tf.artist_name) IN (%s)" % ", ".join("?" * len(artists))
        params = [a.lower() for a in artists]

    exists_filter = "" if rerun else """
        AND NOT EXISTS (
            SELECT 1 FROM tracks_analysis ta
            WHERE ta.track_id = tf.track_id AND ta.artist_id = tf.artist_id
        )"""

    rows = con.execute(f"""
        SELECT tf.track_id, tf.artist_id, tf.album_id,
               tf.track_name, tf.artist_name, tf.album_name,
               tf.album_release_year, tf.lyrics,
               NULL AS isrc, NULL AS artist_isrc
        FROM tracks_flat tf
        WHERE tf.lyrics IS NOT NULL
        {artist_filter}
        {exists_filter}
        ORDER BY tf.artist_id, tf.album_id, tf.track_id
    """, params).fetchall()
    con.close()
    logger.info("%d pistes à analyser", len(rows))
    return [
        {"track_id": r[0], "artist_id": r[1], "album_id": r[2],
         "track_name": r[3], "artist_name": r[4], "album_name": r[5],
         "release_year": r[6], "lyrics": r[7],
         "isrc": r[8], "artist_isrc": r[9]}
        for r in rows
    ]

def _upsert(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> tuple[int, int]:
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

# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(
    artists: list[str] = None,
    db_path: Path = DB_PATH,
    rerun: bool = False,
    level: str = "all",
) -> None:
    _ensure_nltk()
    artists = artists or []

    logger.info("=" * 60)
    logger.info("Lyrics Analyzer — niveaux : %s — artistes : %s",
                level, ", ".join(artists) if artists else "tous")
    logger.info("=" * 60)

    tracks_meta = load_tracks_to_analyze(db_path, artists, rerun)
    if not tracks_meta:
        logger.info("Rien à analyser.")
        return

    # LDA sur corpus complet (sklearn — compatible Python 3.14)
    logger.info("Entraînement LDA sur %d documents…", len(tracks_meta))
    corpus_texts = [_clean_lyrics(m["lyrics"]) for m in tracks_meta]
    lda_model = build_lda(corpus_texts)

    # ── NIVEAU TRACK ──────────────────────────────────────────────────────────
    track_results: list[dict] = []

    if level in ("track", "all"):
        for i, m in enumerate(tracks_meta, 1):
            logger.info("[Track %d/%d] %s — %s", i, len(tracks_meta), m["artist_name"], m["track_name"])
            try:
                row = analyze_track(
                    track_id=m["track_id"], artist_id=m["artist_id"], album_id=m["album_id"],
                    track_name=m["track_name"], artist_name=m["artist_name"], album_name=m["album_name"],
                    raw_lyrics=m["lyrics"], isrc=m["isrc"], artist_isrc=m["artist_isrc"],
                    lda_model=lda_model,
                )
                track_results.append(row)
            except Exception as e:
                logger.error("❌ Track [%s] : %s", m["track_name"], e, exc_info=True)

            # Flush intermédiaire tous les 50
            if len(track_results) > 0 and len(track_results) % 50 == 0:
                con = _init_db(db_path)
                _upsert(con, "tracks_analysis", track_results[-50:])
                con.close()

        # Flush restant
        rem = len(track_results) % 50
        if rem:
            con = _init_db(db_path)
            _upsert(con, "tracks_analysis", track_results[-rem:])
            con.close()
        elif track_results and len(track_results) < 50:
            con = _init_db(db_path)
            _upsert(con, "tracks_analysis", track_results)
            con.close()

        logger.info("✅ tracks_analysis : %d analysées", len(track_results))

    # Si on saute le niveau track, charge depuis BDD
    if level in ("album", "artist") and not track_results:
        logger.info("Chargement tracks_analysis depuis BDD…")
        con = _init_db(db_path)
        af  = ""
        p2: list = []
        if artists:
            af = "WHERE LOWER(artist_name) IN (%s)" % ", ".join("?" * len(artists))
            p2 = [a.lower() for a in artists]
        rows = con.execute(f"SELECT * FROM tracks_analysis {af}", p2).fetchall()
        cols = [d[0] for d in con.description]
        con.close()
        track_results = [dict(zip(cols, row)) for row in rows]
        logger.info("%d tracks chargées", len(track_results))

    # ── NIVEAU ALBUM ──────────────────────────────────────────────────────────
    album_results: list[dict] = []

    if level in ("album", "all"):
        albums_map: dict[tuple, list] = {}
        for tr in tracks_meta:
            if tr.get("album_id"):
                albums_map.setdefault((tr["album_id"], tr["artist_id"]), []).append(tr)

        for (album_id, artist_id), album_tracks in albums_map.items():
            track_ids = {t["track_id"] for t in album_tracks}
            analyzed  = [t for t in track_results if t.get("track_id") in track_ids]
            if not analyzed:
                continue
            sample = album_tracks[0]
            logger.info("[Album] %s — %s (%d titres)",
                        sample["artist_name"], sample["album_name"], len(analyzed))
            try:
                row = aggregate_album(
                    album_id=album_id, artist_id=artist_id,
                    album_name=sample["album_name"], artist_name=sample["artist_name"],
                    release_year=sample.get("release_year"),
                    track_rows=analyzed,
                    raw_lyrics_list=[t["lyrics"] for t in album_tracks],
                    lda_model=lda_model,
                )
                album_results.append(row)
            except Exception as e:
                logger.error("❌ Album [%s] : %s", sample.get("album_name"), e, exc_info=True)

        con = _init_db(db_path)
        ok, bad = _upsert(con, "albums_analysis", album_results)
        con.close()
        logger.info("✅ albums_analysis : %d insérées | %d ignorées", ok, bad)

    # ── NIVEAU ARTIST ─────────────────────────────────────────────────────────
    if level in ("artist", "all"):
        # Charge albums depuis BDD si besoin
        if not album_results:
            con = _init_db(db_path)
            af  = ""
            p3: list = []
            if artists:
                af = "WHERE LOWER(artist_name) IN (%s)" % ", ".join("?" * len(artists))
                p3 = [a.lower() for a in artists]
            rows = con.execute(f"SELECT * FROM albums_analysis {af}", p3).fetchall()
            cols = [d[0] for d in con.description]
            con.close()
            album_results = [dict(zip(cols, row)) for row in rows]

        # Grouper par artiste
        artists_map: dict[int, list] = {}
        for tr in tracks_meta:
            artists_map.setdefault(tr["artist_id"], []).append(tr)

        artist_results: list[dict] = []
        for artist_id, artist_tracks in artists_map.items():
            track_ids  = {t["track_id"] for t in artist_tracks}
            analyzed   = [t for t in track_results if t.get("track_id") in track_ids]
            albums_for = [a for a in album_results if a.get("artist_id") == artist_id]
            sample     = artist_tracks[0]
            logger.info("[Artist] %s — %d titres | %d albums",
                        sample["artist_name"], len(analyzed), len(albums_for))
            try:
                row = aggregate_artist(
                    artist_id=artist_id,
                    artist_name=sample["artist_name"],
                    artist_isrc=sample.get("artist_isrc"),
                    track_rows=analyzed,
                    album_rows=albums_for,
                    raw_lyrics_list=[t["lyrics"] for t in artist_tracks],
                    lda_model=lda_model,
                )
                artist_results.append(row)
            except Exception as e:
                logger.error("❌ Artist [%s] : %s", sample.get("artist_name"), e, exc_info=True)

        con = _init_db(db_path)
        ok, bad = _upsert(con, "artists_analysis", artist_results)
        con.close()
        logger.info("✅ artists_analysis : %d insérées | %d ignorées", ok, bad)

    logger.info("=" * 60)
    logger.info("Pipeline terminé — base : %s", db_path.resolve())
    logger.info("=" * 60)


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