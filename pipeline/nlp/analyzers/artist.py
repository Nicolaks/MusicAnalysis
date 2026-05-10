"""
analyzers/artist.py
===================
Agrégation des métriques NLP au niveau artiste (niveau 3).
Combine tracks_analysis + albums_analysis + analyses propres à la carrière.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.nlp.config import ALL_STOPWORDS
from pipeline.nlp.helpers import avg, clean_lyrics, custom_tokenizer, filtered_tokens, safe_div, std

logger = logging.getLogger(__name__)


def aggregate_artist(
    artist_id: int,
    artist_name: str,
    artist_isrc: Optional[str],
    track_rows: list[dict],
    album_rows: list[dict],
    raw_lyrics_list: list[str],
    lda_model=None,
) -> dict:
    """
    Agrège les métriques de tous les tracks/albums d'un artiste et calcule
    les métriques de carrière (vocabulaire total, centroïde embedding, etc.).
    Retourne un dict prêt pour artists_analysis.
    """
    n_tracks = len(track_rows)
    n_albums = len(album_rows)
    r: dict = {
        "artist_id": artist_id, "artist_name": artist_name,
        "artist_isrc": artist_isrc,
        "album_count": n_albums, "track_count": n_tracks,
    }
    if n_tracks == 0:
        return r

    def _col(key):
        return [t[key] for t in track_rows if t.get(key) is not None]

    # ── 2. Stats lexicales ────────────────────────────────────────────────────
    r["total_word_count"]      = int(sum(_col("word_count")))
    r["avg_word_count"]        = avg(_col("word_count"))
    r["avg_unique_word_count"] = avg(_col("unique_word_count"))
    r["avg_ttr"]               = avg(_col("ttr"))
    r["avg_sentence_length"]   = avg(_col("avg_sentence_length"))

    combined    = " ".join(clean_lyrics(l) for l in raw_lyrics_list)
    all_tokens  = filtered_tokens(combined)
    clean_toks  = filtered_tokens(combined)     # même résultat, conservé pour lisibilité
    freq_all    = FreqDist(clean_toks)
    r["career_vocabulary_size"] = len(set(all_tokens))
    r["career_ttr"]             = safe_div(len(set(all_tokens)), len(all_tokens))
    r["top30_words"] = json.dumps(
        [w for w, _ in freq_all.most_common(30)], ensure_ascii=False
    )

    # ── 3. Sémantique ─────────────────────────────────────────────────────────
    try:
        tfidf = TfidfVectorizer(
            max_features=20,
            stop_words=list(ALL_STOPWORDS),
            tokenizer=custom_tokenizer,
        )
        tfidf.fit_transform([combined])
        r["tfidf_top_keywords"] = json.dumps(
            tfidf.get_feature_names_out().tolist(), ensure_ascii=False
        )
    except Exception:
        r["tfidf_top_keywords"] = None

    if lda_model:
        try:
            topics = lda_model.transform(combined)
            r["lda_topic_distribution"] = json.dumps(
                {str(tid): float(p) for tid, p in topics}
            )
        except Exception:
            r["lda_topic_distribution"] = None
    else:
        r["lda_topic_distribution"] = None

    # Cohérence inter-albums + centroïde PCA 10d
    try:
        from models import get_sbert
        sbert    = get_sbert()
        all_embs = sbert.encode([clean_lyrics(l)[:512] for l in raw_lyrics_list])

        chunk      = max(1, len(raw_lyrics_list) // max(n_albums, 1))
        album_embs = np.array([
            all_embs[i * chunk:(i + 1) * chunk].mean(axis=0)
            for i in range(n_albums)
            if i * chunk < len(all_embs)
        ])
        if len(album_embs) > 1:
            sim_mat = cosine_similarity(album_embs)
            mask    = np.ones(sim_mat.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            r["inter_album_similarity"] = float(sim_mat[mask].mean())
        else:
            r["inter_album_similarity"] = 1.0

        centroid = all_embs.mean(axis=0)
        if len(all_embs) >= 10:
            from sklearn.decomposition import PCA
            pca              = PCA(n_components=10)
            pca.fit(all_embs)
            centroid_reduced = pca.transform(centroid.reshape(1, -1))[0]
        else:
            centroid_reduced = centroid[:10]
        r["career_embedding_centroid"] = json.dumps(centroid_reduced.tolist())

    except Exception as e:
        logger.warning("Embedding artiste échoué : %s", e)
        r["inter_album_similarity"] = r["career_embedding_centroid"] = None

    # ── 4. Stylométrie ────────────────────────────────────────────────────────
    for key in [
        "pos_noun_ratio", "pos_verb_ratio", "pos_adj_ratio",
        "pos_adv_ratio", "pos_pron_ratio",
        "pronoun_i_ratio", "pronoun_we_ratio", "pronoun_you_ratio",
    ]:
        r[f"avg_{key}"] = avg(_col(key))

    style_keys = [
        "pos_noun_ratio", "pos_verb_ratio", "pos_adj_ratio", "pos_pron_ratio",
        "pronoun_i_ratio", "pronoun_we_ratio", "pronoun_you_ratio",
        "rhyme_density", "avg_syllables_line", "ttr", "semantic_density",
    ]
    r["style_signature"] = json.dumps({k: avg(_col(k)) for k in style_keys})

    # ── 5. Rimes & structure ──────────────────────────────────────────────────
    r["avg_rhyme_density"]    = avg(_col("rhyme_density"))
    r["std_rhyme_density"]    = std(_col("rhyme_density"))
    r["avg_syllables_line"]   = avg(_col("avg_syllables_line"))
    r["avg_repetition_ratio"] = avg(_col("repetition_ratio"))
    chorus_vals = [t["chorus_detected"] for t in track_rows if t.get("chorus_detected") is not None]
    r["pct_with_chorus"] = safe_div(sum(chorus_vals), len(chorus_vals)) if chorus_vals else 0.0

    # ── 6. Lisibilité ─────────────────────────────────────────────────────────
    r["avg_flesch_reading_ease"]  = avg(_col("flesch_reading_ease"))
    r["avg_flesch_kincaid_grade"] = avg(_col("flesch_kincaid_grade"))
    r["avg_smog_index"]           = avg(_col("smog_index"))
    r["avg_word_length"]          = avg(_col("avg_word_length"))

    # ── 7. Avancé ─────────────────────────────────────────────────────────────
    r["avg_semantic_density"]  = avg(_col("semantic_density"))
    r["avg_lexical_diversity"] = avg(_col("lexical_diversity"))
    r["avg_hapax_ratio"]       = avg(_col("hapax_ratio"))
    hapax = sum(1 for _, c in freq_all.items() if c == 1)
    r["career_hapax_ratio"] = safe_div(hapax, len(set(all_tokens)))

    return r
