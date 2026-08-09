# AI Resume Screening System

Compares one or more resumes against a job description and predicts
which resume is the best match, with a full explanation (matching
skills, missing skills, similarity %, ranking).

## Quick start — Streamlit UI

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Quick start — custom web frontend (HTML/JS) + REST API

An alternative to Streamlit: a FastAPI backend (`api/main.py`) exposing
the same pipeline over HTTP, plus a standalone HTML/CSS/JS page
(`frontend/index.html`) that calls it — for when you want your own
design instead of the Streamlit look.

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Then just open `frontend/index.html` directly in your browser (double-click
it, or right-click → Open with → your browser). It talks to the API at
`http://localhost:8000` — make sure the `uvicorn` command above is still
running in another terminal.

API docs (interactive, auto-generated): once uvicorn is running, visit
`http://localhost:8000/docs`.

To point the frontend at a different backend URL (e.g. after deploying
the API somewhere), edit the `API_BASE` constant near the top of the
`<script>` block in `frontend/index.html`.

Optional: set `MONGO_URI` to point at a real MongoDB instance to persist
cleaned datasets there (see `database/config.py`). Without it, the
pipeline falls back to local parquet files under `data/processed/`.

## How it works

1. **Cleaning** (`preprocessing/`) — strips URLs/emails/phones/punctuation,
   tokenizes, removes stopwords, lemmatizes (NLTK).
2. **Feature engineering** (`feature_engineering/`) — TF-IDF (always
   available, offline) and Sentence-BERT semantic embeddings (used
   automatically when the model can be downloaded; falls back to TF-IDF
   otherwise — see `embedding_featurizer.py` for details).
3. **Matching** (`models/matcher.py`) — hybrid score:
   `0.6 * semantic_similarity + 0.4 * skill_overlap`. No labeled
   resume↔job pairs exist in the source data, so this is deliberately an
   unsupervised similarity approach rather than a trained classifier.
4. **Role classification** (`models/role_classifier.py`) — a *separate*
   supervised LinearSVC model (chosen after comparing it against
   Logistic Regression and Random Forest via cross-validation — see
   `training/train_role_classifier.py`) predicts which of 42 industry
   categories a resume best fits, using the labels that genuinely exist
   in `training_data.csv`.
5. **Prediction pipeline** (`prediction/prediction_pipeline.py`) — the
   single entry point tying all of the above together; this is what the
   Streamlit app calls.

## Project structure

```
AI_Resume_Screening/
├── data/
│   ├── raw/                 # place source CSV/JSON files here
│   └── processed/           # cleaned parquet outputs (generated)
├── database/                 # reusable MongoDB connector + config
├── preprocessing/            # text cleaning + dataset cleaning
├── feature_engineering/      # TF-IDF, SBERT (+fallback), skill matcher
├── models/                   # HybridMatcher, RoleClassifier
├── training/                 # formal train/test split + tuning script
├── evaluation/                # metrics + confusion matrix report
├── prediction/                # end-to-end pipeline
├── api/                       # FastAPI backend (REST API for a custom frontend)
├── frontend/                  # standalone HTML/CSS/JS page calling the API
├── streamlit_app/             # the UI (app.py)
├── utils/                     # PDF/DOCX text extraction
├── tests/                     # pytest suite
├── saved_models/              # trained artifacts (.joblib)
├── requirements.txt
└── README.md
```

## Results (on the provided datasets)

- **Role classifier**: 99.25% accuracy, 0.99 f1_macro on held-out test
  split (2,000 resumes, 42 categories) — see
  `evaluation/reports/classification_report.txt` and
  `confusion_matrix_top10.png`.
- **person_skills.csv cleaning**: 2,483,376 → 1,873,890 rows after
  removing 609,477 duplicate/near-duplicate rows (24.5%).

## Known limitations / next steps

- The hybrid matcher's weights (0.6 / 0.4) are a reasoned default, not
  tuned against real recruiter feedback — a good next step would be
  collecting a small labeled set of (resume, job, human-judged fit) and
  grid-searching the weights against it.
- SBERT requires downloading `all-MiniLM-L6-v2` from Hugging Face on
  first run; environments with restricted outbound network access will
  automatically use the TF-IDF fallback instead (functional, but less
  semantically aware).
- The skill vocabulary (`data/raw/skills_database.json`) is a curated
  ~115-skill list. Expanding it (e.g. cleaning and merging the raw
  226K-row `06_skills.csv`) would catch more skill mentions, at the cost
  of needing better noise filtering (see Module 1 notes on that file).
- `job_role` (324 classes) was intentionally not used as the classifier
  target due to extreme long-tail imbalance (max 50 samples/class);
  `category` (42 classes) was used instead. A hierarchical
  category → role model is a possible future refinement.
