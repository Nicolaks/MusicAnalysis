import logging
import duckdb

from pathlib import Path
from pipeline.sourcing import genius_explorer
from pipeline.sourcing import isrc_enricher

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
    isrc_enricher.run(db_path=DB_PATH)
    logger.info("Extraction Deezer terminée")
    
def run():
    #run_extract_genuis()
    run_extract_deezer()

if __name__ == "__main__":
    run()
    query("SELECT artist_name, album_name, album_release_date, track_name, track_views FROM tracks_flat ORDER BY RANDOM() LIMIT 100")