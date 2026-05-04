import logging
import duckdb

from pathlib import Path
from pipeline.sourcing import genius_explorer
from pipeline.sourcing import deezer_enricher
from pipeline.sourcing import merge_ranking
from pipeline.sourcing import samples_downloader

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
        
def run():
    #run_extract_genuis()
    #run_extract_deezer()
    #download_samples()
    merge_with_ranking()

if __name__ == "__main__":
    #query("DROP TABLE ranking_data")
    run()
    #query("SELECT artist_name, album_name, album_release_date, track_name, track_views FROM tracks_flat ORDER BY RANDOM() LIMIT 100")
    #query("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'main' ORDER BY table_name, ordinal_position")
    
    #query("SELECT COUNT(*) FROM isrc_data WHERE isrc IS NOT NULL")
    #query("COPY (SELECT * FROM (SUMMARIZE tracks_flat)) TO 'audit_tracks_flat.csv' (HEADER, DELIMITER ',')")
    #query("COPY (SELECT * FROM (SUMMARIZE isrc_data)) TO 'audit_isrc_data.csv' (HEADER, DELIMITER ',')")
    #query("COPY (SELECT * FROM (SUMMARIZE samples_index)) TO 'audit_samples_indexe.csv' (HEADER, DELIMITER ',')")
    query("COPY (SELECT * FROM (SUMMARIZE ranking_data)) TO 'audit_ranking_data.csv' (HEADER, DELIMITER ',')")
    query("COPY (SELECT * FROM ranking_data) TO 'ranking_complet.csv' (HEADER, DELIMITER ',')") 
    #query("SHOW TABLES")
