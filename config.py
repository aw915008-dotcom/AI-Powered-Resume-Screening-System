"""
config.py
---------
Central place for database settings. Reads from environment variables so
credentials never live in source control. Falls back to a local default
for development.
"""

import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "resume_screening")

COLLECTIONS = {
    "resumes": "resumes_clean",
    "job_posts": "job_posts_clean",
    "skills_vocab": "skills_vocabulary",
    "person_skills": "person_skills_clean",
}
