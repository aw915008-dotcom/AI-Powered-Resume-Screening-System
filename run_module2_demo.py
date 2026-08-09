"""
run_module2_demo.py
--------------------
End-to-end demo of Module 2: clean the raw datasets, then store them.

In a real deployment you set MONGO_URI to a running MongoDB instance
(local, Atlas, etc.) and this script inserts the cleaned data there.
If no MongoDB server is reachable (e.g. this sandbox has no DB server
and no network route to one), it automatically falls back to writing
the cleaned data as parquet files under data/processed/ so the rest of
the pipeline (feature engineering, training) can still run against a
consistent interface.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing.data_cleaner import (
    clean_training_data,
    clean_job_postings,
    clean_person_skills,
    clean_skills_vocabulary,
)
from database.mongo_connector import MongoConnector
from database.config import DB_NAME, COLLECTIONS

RAW = Path(__file__).resolve().parent / "data" / "raw"
PROCESSED = Path(__file__).resolve().parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    print("Cleaning datasets...")
    resumes = clean_training_data(RAW / "training_data.csv")
    jobs = clean_job_postings(RAW / "all_job_post.csv")
    person_skills = clean_person_skills(RAW / "05_person_skills.csv")
    skills_vocab = clean_skills_vocabulary(RAW / "06_skills.csv")

    datasets = {
        COLLECTIONS["resumes"]: resumes,
        COLLECTIONS["job_posts"]: jobs,
        COLLECTIONS["person_skills"]: person_skills,
        COLLECTIONS["skills_vocab"]: skills_vocab,
    }

    try:
        db = MongoConnector(db_name=DB_NAME, timeout_ms=3000)
        for collection, df in datasets.items():
            db.insert_dataframe(collection, df)
        db.create_index(COLLECTIONS["resumes"], ["job_role"])
        db.create_index(COLLECTIONS["job_posts"], ["category"])
        db.close()
        print("Stored cleaned data in MongoDB database:", DB_NAME)
    except Exception as e:
        print(f"MongoDB unavailable ({e}); falling back to local parquet storage.")
        for collection, df in datasets.items():
            out_path = PROCESSED / f"{collection}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  saved {len(df):>8} rows -> {out_path}")


if __name__ == "__main__":
    main()
