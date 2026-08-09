"""
matcher.py
----------
The core matching algorithm for this project.

ALGORITHM CHOICE — justified against every option listed in the brief:

  Cosine Similarity / SBERT Similarity   <- CHOSEN (as the base signal)
  XGBoost / Random Forest / Logistic Regression / SVM  <- NOT used here,
      because those are supervised classifiers and this task has no
      labeled (resume, job, match) pairs to train on. Forcing a
      classifier onto this problem would mean inventing synthetic
      labels, which is a data-integrity risk we should not take on a
      real screening tool. (A supervised model IS the right tool for a
      different, well-labeled sub-problem — see role_classifier.py.)

Final design: a HYBRID score, not similarity alone —

    final_score = w_semantic * semantic_similarity + w_skill * skill_overlap

  - semantic_similarity: cosine similarity between SBERT (or TF-IDF
    fallback) embeddings of resume and job description. Captures overall
    meaning/context match.
  - skill_overlap: fraction of the job's required skills explicitly
    present in the resume (from SkillMatcher). Captures hard requirements
    a hiring manager would actually check, and is what makes the
    "matching skills / missing skills" output in the UI possible —
    something a similarity score alone cannot produce.

  Default weights (0.6 semantic, 0.4 skill) favor semantic match while
  still letting concrete skill gaps pull the score down; both are
  exposed as constructor arguments so they can be tuned during
  evaluation (Module 6) against real feedback.
"""

from dataclasses import dataclass, field

from feature_engineering.embedding_featurizer import SemanticFeaturizer
from feature_engineering.skill_matcher import SkillMatcher


@dataclass
class MatchResult:
    final_score: float
    semantic_similarity: float
    skill_overlap_percent: float
    matching_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    extra_skills: list = field(default_factory=list)
    explanation: str = ""


class HybridMatcher:
    def __init__(self, semantic_featurizer: SemanticFeaturizer, skill_matcher: SkillMatcher,
                 weight_semantic: float = 0.6, weight_skill: float = 0.4):
        assert abs(weight_semantic + weight_skill - 1.0) < 1e-6, "weights must sum to 1.0"
        self.semantic = semantic_featurizer
        self.skills = skill_matcher
        self.w_semantic = weight_semantic
        self.w_skill = weight_skill

    def score(self, resume_text_clean: str, job_text_clean: str) -> MatchResult:
        # Semantic similarity
        r_emb = self.semantic.encode([resume_text_clean])
        j_emb = self.semantic.encode([job_text_clean])
        sem_sim = float(self.semantic.similarity(r_emb, j_emb)[0][0])
        sem_sim = max(0.0, min(1.0, sem_sim))  # clip — TF-IDF fallback sim is already in [0,1], SBERT cosine can dip slightly negative

        # Skill overlap
        skill_result = self.skills.compare(resume_text_clean, job_text_clean)
        skill_overlap = skill_result["skill_overlap_percent"] / 100.0

        final = self.w_semantic * sem_sim + self.w_skill * skill_overlap

        explanation = self._build_explanation(sem_sim, skill_result)

        return MatchResult(
            final_score=round(final * 100, 1),
            semantic_similarity=round(sem_sim * 100, 1),
            skill_overlap_percent=skill_result["skill_overlap_percent"],
            matching_skills=skill_result["matching_skills"],
            missing_skills=skill_result["missing_skills"],
            extra_skills=skill_result["extra_skills"],
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(sem_sim: float, skill_result: dict) -> str:
        parts = []
        if sem_sim >= 0.7:
            parts.append("strong overall semantic match with the job description")
        elif sem_sim >= 0.4:
            parts.append("moderate semantic overlap with the job description")
        else:
            parts.append("low semantic overlap with the job description")

        if skill_result["missing_skills"]:
            parts.append(f"missing {len(skill_result['missing_skills'])} required skill(s): "
                          f"{', '.join(skill_result['missing_skills'][:5])}")
        else:
            parts.append("covers all identified required skills")

        return "; ".join(parts)

    def rank(self, resumes: dict, job_text_clean: str) -> list:
        """resumes: {resume_id: resume_text_clean}. Returns list of
        (resume_id, MatchResult) sorted best-first."""
        results = [(rid, self.score(text, job_text_clean)) for rid, text in resumes.items()]
        return sorted(results, key=lambda x: x[1].final_score, reverse=True)
