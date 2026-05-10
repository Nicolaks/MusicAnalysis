"""
analyzers/album.py
==================
Agrégation des métriques NLP au niveau album (niveau 2).
Combine les résultats de tracks_analysis + analyses propres à l'album.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.nlp.config import ALL_STOPWORDS
from pipeline.nlp.helpers import avg, clean_lyrics, custom_tokenizer, filtered_tokens, safe_div,dominant_emotions


# ── 8. Émotions & champs lexicaux ────────────────────────────────────────────
def _avg_json_scores(rows: list[dict], key: str) -> dict[str, float]:
    """Moyenne des scores JSON sur un ensemble de tracks."""
    all_scores: list[dict] = []
    for row in rows:
        raw = row.get(key)
        if raw:
            try:
                all_scores.append(json.loads(raw))
            except Exception:
                pass
    if not all_scores:
        return {}
    keys = all_scores[0].keys()
    return {k: avg([s[k] for s in all_scores if k in s]) for k in keys}

def aggregate_album(
    album_id: int,
    artist_id: int,
    album_name: str,
    artist_name: str,
    release_year: Optional[int],
    track_rows: list[dict],
    raw_lyrics_list: list[str],
    lda_model=None,
) -> dict:
    """
    Agrège les métriques des pistes d'un album et calcule les métriques
    propres à l'album (TTR global, cohérence thématique, etc.).
    Retourne un dict prêt pour albums_analysis.
    """
    n = len(track_rows)
    r: dict = {
        "album_id": album_id, "artist_id": artist_id,
        "album_name": album_name, "artist_name": artist_name,
        "release_year": release_year, "track_count": n,
    }
    if n == 0:
        return r

    def _col(key):
        return [t[key] for t in track_rows if t.get(key) is not None]

    # ── 2. Stats lexicales ────────────────────────────────────────────────────
    r["total_word_count"]      = int(sum(_col("word_count")))
    r["avg_word_count"]        = avg(_col("word_count"))
    r["avg_unique_word_count"] = avg(_col("unique_word_count"))
    r["avg_ttr"]               = avg(_col("ttr"))
    r["avg_sentence_length"]   = avg(_col("avg_sentence_length"))

    combined   = " ".join(clean_lyrics(l) for l in raw_lyrics_list)
    all_tokens = filtered_tokens(combined)
    r["album_vocabulary_size"] = len(set(all_tokens))
    r["album_ttr"]             = safe_div(len(set(all_tokens)), len(all_tokens))
    freq_all   = FreqDist(all_tokens)
    r["top20_words"] = json.dumps(
        [w for w, _ in freq_all.most_common(20)], ensure_ascii=False
    )

    # ── 3. Sémantique ─────────────────────────────────────────────────────────
    try:
        tfidf = TfidfVectorizer(
            max_features=15,
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
            best   = max(topics, key=lambda x: x[1])
            r["lda_dominant_topic"]     = int(best[0])
            r["lda_topic_distribution"] = json.dumps(
                {str(tid): float(p) for tid, p in topics}
            )
        except Exception:
            r["lda_dominant_topic"] = r["lda_topic_distribution"] = None
    else:
        r["lda_dominant_topic"] = r["lda_topic_distribution"] = None

    try:
        from pipeline.nlp.models import get_sbert
        sbert = get_sbert()
        embs  = sbert.encode([clean_lyrics(l)[:512] for l in raw_lyrics_list])
        if len(embs) > 1:
            sim_mat = cosine_similarity(embs)
            mask    = np.ones(sim_mat.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            r["intra_album_similarity"] = float(sim_mat[mask].mean())
        else:
            r["intra_album_similarity"] = 1.0
    except Exception:
        r["intra_album_similarity"] = None

    # ── 4. Stylométrie (moyennes) ─────────────────────────────────────────────
    for key in [
        "pos_noun_ratio", "pos_verb_ratio", "pos_adj_ratio",
        "pos_adv_ratio", "pos_pron_ratio",
        "pronoun_i_ratio", "pronoun_we_ratio", "pronoun_you_ratio",
    ]:
        r[f"avg_{key}"] = avg(_col(key))

    # ── 5. Rimes & structure ──────────────────────────────────────────────────
    r["avg_rhyme_density"]    = avg(_col("rhyme_density"))
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
    r["album_hapax_ratio"] = safe_div(hapax, len(set(all_tokens)))
    
    avg_emotion = _avg_json_scores(track_rows, "emotion_scores")
    avg_lex     = _avg_json_scores(track_rows, "lexical_field_scores")

    r["avg_emotion_scores"]       = json.dumps(avg_emotion, ensure_ascii=False)
    r["dominant_emotions"]        = json.dumps(dominant_emotions(avg_emotion), ensure_ascii=False)
    r["avg_lexical_field_scores"] = json.dumps(avg_lex, ensure_ascii=False)
    r["dominant_lexical_fields"]  = json.dumps(dominant_emotions(avg_lex), ensure_ascii=False)

    return r
