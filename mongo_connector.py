"""
mongo_connector.py
-------------------
Thin, reusable wrapper around PyMongo. Every other module talks to Mongo
through this class instead of instantiating MongoClient directly, so we
have one place to change pooling/timeout/retry behaviour.

Usage
-----
    from database.mongo_connector import MongoConnector
    from database.config import DB_NAME, COLLECTIONS

    db = MongoConnector(db_name=DB_NAME)
    db.insert_dataframe(COLLECTIONS["resumes"], cleaned_resumes_df)
    docs = db.find(COLLECTIONS["resumes"], {"job_role": "Data Scientist"})
    db.close()
"""

import logging
from typing import Iterable, Optional

import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, PyMongoError

from database.config import MONGO_URI, DB_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class MongoConnector:
    """Reusable MongoDB access layer.

    Deliberately small and boring: connect, insert, find, upsert, index,
    close. Anything fancier (aggregation pipelines for feature
    engineering) belongs in the module that needs it, built on top of
    `self.db` directly.
    """

    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME, timeout_ms: int = 5000):
        self.uri = uri
        self.db_name = db_name
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
            self.client.admin.command("ping")  # fail fast if unreachable
            self.db = self.client[db_name]
            logger.info(f"Connected to MongoDB database '{db_name}'")
        except ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB at {uri}: {e}")
            raise

    def insert_dataframe(self, collection: str, df: pd.DataFrame, clear_first: bool = True) -> int:
        """Insert a cleaned pandas DataFrame as documents. Returns count inserted."""
        if clear_first:
            self.db[collection].delete_many({})
        records = df.to_dict(orient="records")
        if not records:
            logger.warning(f"No records to insert into '{collection}'")
            return 0
        try:
            result = self.db[collection].insert_many(records)
            logger.info(f"Inserted {len(result.inserted_ids)} docs into '{collection}'")
            return len(result.inserted_ids)
        except PyMongoError as e:
            logger.error(f"Insert failed for '{collection}': {e}")
            raise

    def find(self, collection: str, query: Optional[dict] = None, projection: Optional[dict] = None) -> list:
        query = query or {}
        return list(self.db[collection].find(query, projection))

    def upsert(self, collection: str, match: dict, update: dict) -> None:
        self.db[collection].update_one(match, {"$set": update}, upsert=True)

    def create_index(self, collection: str, fields: Iterable[str]) -> None:
        self.db[collection].create_index([(f, ASCENDING) for f in fields])
        logger.info(f"Created index on {collection}: {list(fields)}")

    def count(self, collection: str) -> int:
        return self.db[collection].count_documents({})

    def close(self) -> None:
        self.client.close()
        logger.info("MongoDB connection closed")
