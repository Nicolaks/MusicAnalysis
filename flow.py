"""
flow.py
=======
Orchestrateur principal du pipeline de données musicales. 
Gère l'enchaînement des étapes d'extraction, d'enrichissement, d'analyse 
et la génération de rapports d'audit.

Flux global :
    1. genius_explorer     : Extraction des catalogues artistes depuis Genius.
    2. deezer_enricher     : Récupération des ISRC et métadonnées via Deezer[cite: 1].
    3. samples_downloader  : Téléchargement des extraits audio (mp3).
    4. merge_ranking       : Consolidation des données de classement.
    5. kworb_streams       : Récupération des statistiques de streaming Spotify[cite: 1].
    6. audio_features      : Analyse technique des fichiers audio locaux.
    7. Audit (CSV)         : Exportation des tables et résumés statistiques.
"""

import logging
import duckdb

from pathlib import Path
from pipeline.sourcing import genius_explorer
from pipeline.sourcing import deezer_enricher
from pipeline.sourcing import merge_ranking
from pipeline.sourcing import samples_downloader
from pipeline.sourcing import kworb_streams
from pipeline.analyse import audio_features
from pipeline.nlp import lyrics_analyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler("logs/flow.log", encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

DB_PATH = Path("data/warehouse.duckdb")

ARTISTS = [
    "PLK",
    "SDM",
    "DAMSO",
    "Ninho",
    "Orelsan",
    "Booba",
    "SCH",
    "PNL",
    "NISKA",
    "Theodora",
    "Bad Bunny",
]

# ─────────────────────────────────────────────
# Utilitaires de Base de Données
# ─────────────────────────────────────────────

def query(sql: str) -> None:
    """Exécute une requête SQL sur DuckDB et affiche le résultat en console."""
    logger.info("Connexion à %s", DB_PATH)
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute(sql).df()
    con.close()
    logger.info("%d lignes retournées", len(df))
    print(df.to_string(index=False))
    
def get_db_table_schema():
    logger.info("Génération du fichier CSV des tables de la BDD")
    query(
        """
        COPY (
            SELECT 
                table_name,
                column_name,
                data_type
            FROM information_schema.columns
            ORDER BY table_name, ordinal_position
        )
        TO 'rapports/schema_all_tables.csv'
        WITH (HEADER, DELIMITER ',')
    """)
    logger.info("Génération terminées")
    
def run_extract_genuis():
    """Étape 1 : Explore Genius pour lister les morceaux de chaque artiste."""
    logger.info("Lancement du pipeline pour %d artiste(s)", len(ARTISTS))
    for i, artist in enumerate(ARTISTS, start=1):
        logger.info(">>> [%d/%d] %s", i, len(ARTISTS), artist)
        genius_explorer.run(artists=[artist], db_path=DB_PATH)
    logger.info("Pipeline terminé")
    
def run_extract_deezer():
    """Étape 2 : Enrichit les morceaux avec les ISRC et métadonnées Deezer."""
    logger.info("Lancement de l'extraction Deezer")
    deezer_enricher.run(db_path=DB_PATH)
    logger.info("Extraction Deezer terminée")
    
def download_samples():
    """Étape 3 : Télécharge les previews audio pour permettre l'analyse ultérieure."""
    logger.info("Lancement du téléchargement des extraits pour analyse")
    samples_downloader.run(db_path=DB_PATH)
    logger.info("Téléchargement terminé")
    
def merge_with_ranking():
    """Étape 4 : Fusionne les données locales avec les rankings externes."""
    logger.info("Lancement du merge sur les ranking")
    merge_ranking.run(db_path=DB_PATH)
    logger.info("Merge Ranking terminé")
      
def get_streams():
    """Étape 5 : Récupère les données de streams Spotify via le scraping Kworb."""
    logger.info("Lancement de la recherche des ID Spotify pour obtenir les streams Kworb")
    kworb_streams.run(db_path=DB_PATH)
    logger.info("Recherche terminée")
    
def audio_features_analysis():
    """Étape 6 : Analyse les fichiers mp3 téléchargés (bpm, énergie, etc.)."""
    logger.info("Début de l'analyse des features audio")
    audio_features.run(db_path=DB_PATH)
    logger.info("Analyse des features audio terminée")
    
def lyrics_analyser_process():
    """Étape 7 : Analyse des paroles afin de récupérer les métriques utiles."""
    logger.info("Début de l'analyse des paroles")
    lyrics_analyzer.run(db_path=DB_PATH, artists=ARTISTS)
    logger.info("Analyse des paroles terminée")
    
# ─────────────────────────────────────────────
# Orchestration et Audit
# ─────────────────────────────────────────────

def get_audit_csv(enable: bool): 
    """
    Génère des exports CSV pour chaque table du warehouse afin d'auditer 
    la qualité et la complétude des données.
    """  
    audit_table_list = [
        "audio_features_local",
        "kworb_streams",
        "ranking_data",
        "samples_index",
        "isrc_data",
        "tracks_flat",
        "tracks_analysis",
        "albums_analysis",
        "artists_analysis",
    ]    
    if enable:
        for table in audit_table_list:
            logger.info("Génération du rapport pour la table : {table}")
            query(f"COPY (SELECT * FROM {table}) TO 'rapports/audit_{table}.csv' (HEADER, DELIMITER ',')")
            query(f"COPY (SELECT * FROM (SUMMARIZE {table})) TO 'rapports/summarize_audit_{table}.csv' (HEADER, DELIMITER ',')")
        logger.info("Les rapports ont étés générés dans /rapports")
       
def run():
    """Exécute l'intégralité du pipeline dans l'ordre logique des dépendances."""
    run_extract_genuis()
    run_extract_deezer()
    download_samples()
    merge_with_ranking()
    get_streams()
    audio_features_analysis()
    lyrics_analyser_process()
    

if __name__ == "__main__":
    """Point d'entrée : Exécution du flow complet puis génération de l'audit"""
    #query("DROP TABLE kworb_streams")
    #query("DROP TABLE tracks_analysis")
    #query("DROP TABLE albums_analysis")
    #query("DROP TABLE artists_analysis")
    run()
    #query("SHOW TABLES")
    get_audit_csv(True)
    get_db_table_schema()
