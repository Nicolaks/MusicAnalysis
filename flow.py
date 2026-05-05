import logging
import duckdb

from pathlib import Path
from pipeline.sourcing import genius_explorer
from pipeline.sourcing import deezer_enricher
from pipeline.sourcing import merge_ranking
from pipeline.sourcing import samples_downloader
from pipeline.sourcing import kworb_streams
from pipeline.analyse import audio_features

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
]

def query(sql: str) -> None:
    logger.info("Connexion à %s", DB_PATH)
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute(sql).df()
    con.close()
    logger.info("%d lignes retournées", len(df))
    print(df.to_string(index=False))
    
def run_extract_genuis():
    logger.info("Lancement du pipeline pour %d artiste(s)", len(ARTISTS))
    for i, artist in enumerate(ARTISTS, start=1):
        logger.info(">>> [%d/%d] %s", i, len(ARTISTS), artist)
        genius_explorer.run(artists=[artist], db_path=DB_PATH)
    logger.info("Pipeline terminé")
    
def run_extract_deezer():
    logger.info("Lancement de l'extraction Deezer")
    deezer_enricher.run(db_path=DB_PATH)
    logger.info("Extraction Deezer terminée")
    
def merge_with_ranking():
    logger.info("Lancement du merge sur les ranking")
    merge_ranking.run(db_path=DB_PATH)
    logger.info("Merge Ranking terminé")
    
def download_samples():
    logger.info("Lancement du téléchargement des extraits pour analyse")
    samples_downloader.run(db_path=DB_PATH)
    logger.info("Téléchargement terminé")
    
def get_streams():
    logger.info("Lancement de la recherche des ID Spotify pour obtenir les streams Kworb")
    kworb_streams.run(db_path=DB_PATH)
    logger.info("Recherche terminée")
    
def audio_features_analysis():
    logger.info("Début de l'analyse des features audio")
    audio_features.run(db_path=DB_PATH)
    logger.info("Analyse des features audio terminée")
        
def run():
    run_extract_genuis()
    run_extract_deezer()
    download_samples()
    merge_with_ranking()
    get_streams()
    audio_features_analysis()
    
def get_audit_csv(enable: bool):   
    audit_table_list = [
        "audio_features_local",
        "kworb_streams",
        "ranking_data",
        "samples_index",
        "isrc_data",
        "tracks_flat",
    ]    
    if enable:
        for table in audit_table_list:
            logger.info("Génération du rapport pour la table : {table}")
            query(f"COPY (SELECT * FROM {table}) TO 'rapports/audit_{table}.csv' (HEADER, DELIMITER ',')")
            query(f"COPY (SELECT * FROM (SUMMARIZE {table})) TO 'rapports/summarize_audit_{table}.csv' (HEADER, DELIMITER ',')")
        logger.info("Les rapports ont étés générés dans /rapports")

if __name__ == "__main__":
    #query("DROP TABLE kworb_streams")
    run()
    #query("SHOW TABLES")
    get_audit_csv(True)
