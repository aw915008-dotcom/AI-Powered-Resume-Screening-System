"""
app.py
------
Module 8 — Streamlit UI for the AI Resume Screening System.

Run with:  streamlit run streamlit_app/app.py   (from the project root)

Layout:
  Sidebar : upload resumes (PDF/DOCX/TXT), paste job description, run button
  Main    : per-resume scores, winner, similarity %, matching/missing
            skills, a comparison chart, and a downloadable text report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from prediction.prediction_pipeline import ResumeScreeningPipeline
from utils.file_parser import extract_text

st.set_page_config(page_title="AI Resume Screening", layout="wide")


@st.cache_resource
def load_pipeline():
    return ResumeScreeningPipeline(role_classifier_path=str(Path(__file__).resolve().parent.parent / "saved_models" / "role_classifier_tuned.joblib"))


def build_report_text(result) -> str:
    lines = [f"AI Resume Screening Report — {result.job_title}", "=" * 50, ""]
    lines.append(f"Winner: {result.winner_id} ({result.winner_reason})\n")
    for rank, r in enumerate(result.rankings, start=1):
        lines.append(f"#{rank} — {r.resume_id}")
        lines.append(f"  Final score:        {r.final_score}%")
        lines.append(f"  Semantic similarity: {r.semantic_similarity}%")
        lines.append(f"  Skill overlap:       {r.skill_overlap_percent}%")
        lines.append(f"  Predicted category:  {r.predicted_category} ({r.category_confidence}%)")
        lines.append(f"  Matching skills:     {', '.join(r.matching_skills) or '-'}")
        lines.append(f"  Missing skills:      {', '.join(r.missing_skills) or '-'}")
        lines.append(f"  Explanation:         {r.explanation}")
        lines.append("")
    return "\n".join(lines)


def main():
    st.title("AI Resume Screening System")
    st.caption("Upload resumes, paste a job description, and get a ranked, explainable match.")

    with st.sidebar:
        st.header("1. Job description")
        job_title = st.text_input("Job title", value="Untitled role")
        job_description = st.text_area("Paste the job description", height=220)

        st.header("2. Resumes")
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True
        )

        run_clicked = st.button("Run comparison", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Fill in a job description and upload at least one resume, then click **Run comparison**.")
        return

    if not job_description.strip():
        st.error("Please paste a job description.")
        return
    if not uploaded_files:
        st.error("Please upload at least one resume.")
        return

    resumes = {}
    for f in uploaded_files:
        try:
            text = extract_text(f.name, f.read())
            if text.strip():
                resumes[f.name] = text
            else:
                st.warning(f"Could not extract text from {f.name} (empty result) — skipped.")
        except Exception as e:
            st.warning(f"Failed to read {f.name}: {e}")

    if not resumes:
        st.error("No readable resumes.")
        return

    with st.spinner("Scoring resumes..."):
        pipeline = load_pipeline()
        result = pipeline.run(job_description, resumes, job_title=job_title)

    st.success(f"Winner: **{result.winner_id}** — {result.winner_reason}")

    # --- Comparison chart ---
    df = pd.DataFrame([{
        "Resume": r.resume_id,
        "Final score": r.final_score,
        "Semantic similarity": r.semantic_similarity,
        "Skill overlap": r.skill_overlap_percent,
    } for r in result.rankings])

    fig = go.Figure()
    fig.add_bar(name="Final score", x=df["Resume"], y=df["Final score"])
    fig.add_bar(name="Semantic similarity", x=df["Resume"], y=df["Semantic similarity"])
    fig.add_bar(name="Skill overlap", x=df["Resume"], y=df["Skill overlap"])
    fig.update_layout(barmode="group", yaxis_title="%", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    # --- Per-resume detail cards ---
    for rank, r in enumerate(result.rankings, start=1):
        with st.expander(f"#{rank} — {r.resume_id} ({r.final_score}%)", expanded=(rank == 1)):
            c1, c2, c3 = st.columns(3)
            c1.metric("Final score", f"{r.final_score}%")
            c2.metric("Semantic similarity", f"{r.semantic_similarity}%")
            c3.metric("Skill overlap", f"{r.skill_overlap_percent}%")

            st.markdown(f"**Predicted category:** {r.predicted_category} ({r.category_confidence}% confidence)")
            st.markdown(f"**Explanation:** {r.explanation}")

            colm, colx = st.columns(2)
            with colm:
                st.markdown("**Matching skills**")
                st.write(", ".join(r.matching_skills) or "—")
            with colx:
                st.markdown("**Missing skills**")
                st.write(", ".join(r.missing_skills) or "—")

    # --- Download report ---
    report_text = build_report_text(result)
    st.download_button(
        "Download report (.txt)", data=report_text,
        file_name=f"resume_screening_report_{job_title.replace(' ', '_')}.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
