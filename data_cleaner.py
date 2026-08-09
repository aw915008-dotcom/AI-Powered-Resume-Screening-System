"""
data_cleaner.py
----------------
Dataset-level cleaning: deduplication, missing-value handling, column
pruning, and orchestration of `text_cleaner.clean_text` over the text
columns that matter for matching (resume text / job description).

Two datasets are treated as first-class citizens for the matching engine:
  - training_data.csv   -> resumes (label = Job Role / Category)
  - all_job_post.csv    -> job postings

The large people/education/experience/skills tables (01-06) are treated
as a *skills knowledge base* used later for skill-matching, not as
training rows themselves — they have no resume<->job pairing, so they
can't directly supervise a matcher. See Module 1 notes.
"""

import logging
from pathlib import Path

import pandas as pd

from preprocessing.text_cleaner import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans one tabular dataset at a time. One instance per file keeps
    logging and stats scoped and makes the class easy to unit test."""

    def __init__(self, name: str):
        self.name = name
        self.stats = {}

    def _log_stats(self, df: pd.DataFrame, stage: str):
        logger.info(f"[{self.name}] after {stage}: {df.shape[0]} rows, {df.shape[1]} cols")

    def drop_duplicates(self, df: pd.DataFrame, subset=None) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        removed = before - len(df)
        self.stats["duplicates_removed"] = removed
        logger.info(f"[{self.name}] removed {removed} duplicate rows")
        return df

    def drop_rows_with_missing(self, df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
        """Drop rows missing a REQUIRED field (e.g. resume text). Optional
        fields (email, linkedin...) are left as NaN rather than dropped,
        since dropping on optional fields would throw away most of the
        dataset for no benefit to the matching task."""
        before = len(df)
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        removed = before - len(df)
        self.stats["missing_required_dropped"] = removed
        logger.info(f"[{self.name}] dropped {removed} rows missing required fields {required_cols}")
        return df

    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        cols_present = [c for c in columns if c in df.columns]
        return df.drop(columns=cols_present)

    def clean_text_column(self, df: pd.DataFrame, column: str, new_column: str = None) -> pd.DataFrame:
        new_column = new_column or f"{column}_clean"
        df[new_column] = df[column].apply(clean_text)
        return df


def clean_training_data(path: str) -> pd.DataFrame:
    """Clean training_data.csv (resumes)."""
    cleaner = DataCleaner("training_data")
    df = pd.read_csv(path)
    cleaner._log_stats(df, "load")

    df = cleaner.drop_duplicates(df, subset=["Resume ID"])
    df = cleaner.drop_rows_with_missing(df, required_cols=["Resume Text", "Job Role"])
    df = cleaner.clean_text_column(df, "Resume Text", "resume_text_clean")

    # Normalize the pipe-delimited Skills column into a real list column.
    df["skills_list"] = df["Skills"].fillna("").apply(
        lambda s: [x.strip() for x in s.split("|") if x.strip()]
    )

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    cleaner._log_stats(df, "full cleaning")
    return df


def clean_job_postings(path: str) -> pd.DataFrame:
    """Clean all_job_post.csv (job descriptions)."""
    cleaner = DataCleaner("all_job_post")
    df = pd.read_csv(path)
    cleaner._log_stats(df, "load")

    df = cleaner.drop_duplicates(df, subset=["job_id"])
    df = cleaner.drop_rows_with_missing(df, required_cols=["job_description", "job_title"])
    df = cleaner.clean_text_column(df, "job_description", "job_description_clean")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    cleaner._log_stats(df, "full cleaning")
    return df


def clean_person_skills(path: str) -> pd.DataFrame:
    """Clean 05_person_skills.csv — this file alone had 23.7% duplicate
    rows (see Module 1), so dedup is the dominant operation here."""
    cleaner = DataCleaner("person_skills")
    df = pd.read_csv(path)
    cleaner._log_stats(df, "load")

    df = cleaner.drop_rows_with_missing(df, required_cols=["skill"])
    df = cleaner.drop_duplicates(df, subset=["person_id", "skill"])
    df["skill"] = df["skill"].str.strip().str.lower()
    df = cleaner.drop_duplicates(df, subset=["person_id", "skill"])  # re-dedupe after normalizing case

    cleaner._log_stats(df, "full cleaning")
    return df


def clean_skills_vocabulary(path: str) -> pd.DataFrame:
    """Clean 06_skills.csv into a canonical, deduplicated skills vocabulary."""
    cleaner = DataCleaner("skills_vocab")
    df = pd.read_csv(path)
    df = cleaner.drop_rows_with_missing(df, required_cols=["skill"])
    df["skill"] = df["skill"].str.strip()
    df = cleaner.drop_duplicates(df, subset=["skill"])
    cleaner._log_stats(df, "full cleaning")
    return df


if __name__ == "__main__":
    # Quick smoke test when run directly: `python -m preprocessing.data_cleaner`
    RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
    OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
    OUT.mkdir(parents=True, exist_ok=True)

    resumes = clean_training_data(RAW / "training_data.csv")
    jobs = clean_job_postings(RAW / "all_job_post.csv")

    resumes.to_parquet(OUT / "resumes_clean.parquet", index=False)
    jobs.to_parquet(OUT / "jobs_clean.parquet", index=False)
    print("Saved cleaned datasets to", OUT)
