"""
skill_matcher.py
-----------------
Deterministic, rule-based skill matching — separate from the semantic
similarity score on purpose.

The prediction pipeline (Module 7) must output concrete lists like
"matching skills" and "missing skills". A cosine-similarity score alone
cannot produce that; it needs an explicit skill vocabulary lookup. This
module extracts skill mentions from free text against the canonical
skills vocabulary built in Module 2 (06_skills.csv / skills_list.csv)
and computes set-based overlap.
"""

import re
from typing import Iterable


class SkillMatcher:
    def __init__(self, skills_vocabulary: Iterable[str]):
        # Longest-skill-first so multi-word skills (e.g. "machine learning")
        # are matched before their substrings ("learning") could shadow them.
        self.vocab = sorted({s.strip().lower() for s in skills_vocabulary if s and s.strip()}, key=len, reverse=True)
        self._patterns = [
            (skill, re.compile(r"(?<!\w)" + re.escape(skill) + r"(?!\w)"))
            for skill in self.vocab
        ]

    def extract_skills(self, text: str) -> set:
        text = (text or "").lower()
        found = set()
        for skill, pattern in self._patterns:
            if pattern.search(text):
                found.add(skill)
        return found

    def compare(self, resume_text: str, job_text: str) -> dict:
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(job_text)

        matching = resume_skills & job_skills
        missing = job_skills - resume_skills
        extra = resume_skills - job_skills

        overlap_pct = (len(matching) / len(job_skills) * 100) if job_skills else 0.0

        return {
            "resume_skills": sorted(resume_skills),
            "job_skills": sorted(job_skills),
            "matching_skills": sorted(matching),
            "missing_skills": sorted(missing),
            "extra_skills": sorted(extra),
            "skill_overlap_percent": round(overlap_pct, 1),
        }
