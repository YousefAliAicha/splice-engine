# Splice — Adaptive Movie Recommender Engine

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-red)
![Datasets](https://img.shields.io/badge/datasets-MovieLens%20100k%20%2F%201M-orange)

A movie recommender that switches strategy based on how much is known about a user — from a genre-weighted cold-start fallback, to item-based KNN for sparse histories, to a stacked ensemble of five collaborative-filtering models blended by a LightGBM meta-learner for established users. Wrapped in a Streamlit dashboard that collects real feedback and closes the loop with its own analysis pipeline.

Built by Yousef Ali Aicha as a portfolio project — systems-level thinking applied to a well-known ML problem, not just a model-fitting exercise.

---

## Features

**Recommendation engines**

- **Cold start (0 ratings)** — genre-affinity scoring against the catalog, weighted by how well-liked a movie is overall
- **Sparse users (1–4 ratings)** — item-based KNN, cosine similarity over a movie × user rating matrix
- **Established users (5+ ratings)** — stacked ensemble: SVD, item-based KNN-with-means, Baseline, SlopeOne, and Co-Clustering generate out-of-fold predictions, blended by a LightGBM meta-learner
- **Hybrid mode (optional)** — collaborative ensemble score blended with a content-based signal from synopsis embeddings (sentence-transformers + FAISS), for users the ensemble already has a real prediction for

**Dashboard** (Streamlit, served locally)

- Curated taste profiles ("Action & Adventure," "Romance," etc.) — no login, just pick a mood
- A short multi-step quiz that builds a genre-affinity vector and scores the catalog against it
- TMDB posters and synopses on every recommendation
- "More like this (by plot)" — genuine content-based similarity via synopsis embeddings, distinct from the collaborative KNN engine
- Per-recommendation 0–10 rating slider, logged to a feedback CSV alongside which engine served it and what it predicted
- End-of-quiz experience rating, tracked over time
- Live model diagnostics panel: Test RMSE via `st.metric`, meta-learner feature importances
- "Advanced" panel — direct access to the full ensemble by user ID, plus a toggle to compare pure-collaborative vs. hybrid output side by side

**Feedback & analysis pipeline**

- Every rating gets logged with a timestamp, engine, predicted score (where one exists), and the user's actual rating
- `analysis/feedback_analysis.py` — calibration (predicted vs. actual), engine comparison, rating distribution, quiz-experience trend over time, all via matplotlib
- `analysis/generate_demo_data.py` — generates plausible synthetic feedback correlated with real trained predictions, for demo/portfolio purposes, kept entirely separate from real usage data

**Engineering discipline**

- Leakage-safe evaluation: 10-fold out-of-fold cross-validation for the meta-learner, held-out test set touched exactly once
- `tests/test_engines.py` — routing logic, cold-start fallbacks, candidate generation; caught two real bugs during development (see [Engineering Decisions](#engineering-decisions))
- Dataset-agnostic `src/` package — the same model code trains on either MovieLens 100k or 1M via a `--dataset` flag, with results saved separately per dataset
- `.env`-based secrets handling for the TMDB API key, never committed

---

## Benchmarks

| Dataset | Test RMSE | Users | Movies | Ratings |
|---|---|---|---|---|
| MovieLens 100k | 0.9010 | 943 | 1,682 | 100,000 |
| MovieLens 1M | 0.8492 | 6,040 | 3,706 | 1,000,209 |

Base model out-of-fold RMSEs (MovieLens 100k):

| Model | RMSE |
|---|---|
| item-based KNN | 0.9294 |
| Baseline | 0.9464 |
| SlopeOne | 0.9493 |
| Co-Clustering | 0.9708 |
| SVD | 0.9725 |
| **Meta-learner (blended)** | **0.8684** |

These are untuned baselines — hyperparameters were carried over from initial development rather than searched. A tuning pass is noted as a planned next step; see [Engineering Decisions](#engineering-decisions).

---

## How It Works

### Cold start

A collaborative-filtering ensemble is only as good as the rating history it has to work with. A brand-new user — which is every visitor to the quiz or the taste-profile browser — has none, so scoring them through SVD, item-KNN, or the meta-learner isn't just weaker, it's meaningless: those models have no signal for a user_id they've never seen.

Splice routes on rating count rather than forcing every user through one model:

| Ratings on file | Engine used | Why |
|---|---|---|
| 0 | Genre-affinity scoring | No collaborative signal exists yet. Falls back to content (genre) matching weighted by how well-liked a movie is overall. |
| 1–4 | Item-based KNN | Too little history for the ensemble's stat features (user bias, std, count) to be reliable, but enough for "movies similar to the one you just rated highly" to work. |
| 5+ | Full stacked ensemble | Enough history for user-level bias and the meta-learner's blending to actually mean something. |

The quiz and taste-profile browser are both cold-start by definition — a first-time visitor has no account and no history — so they run entirely on the genre-affinity path (`src/engines.py::score_by_genre_match`). This is the same fallback the router uses internally for any real user with zero ratings (`src/engines.py::recommend`), not a separate simplified demo path. The dashboard states this explicitly under quiz/profile results rather than presenting genre-matched picks as if they came from the trained ensemble.

The "Advanced" panel exists specifically to demonstrate the full ensemble against a user who *does* have rating history, since that's not something a first-time visitor can trigger on their own.

### The stacked ensemble

Five collaborative-filtering models (SVD, item-KNN-with-means, Baseline, SlopeOne, Co-Clustering) each produce out-of-fold predictions via 10-fold cross-validation — every training row is scored by a model that never saw that row during training, which is what keeps the meta-learner from just learning to trust whichever base model overfit hardest. Those predictions, plus per-user and per-item statistical features (bias, std, rating count), feed a LightGBM regressor that learns how to weight them.

**SVD++ was tested and dropped.** It's usually one of the stronger models on MovieLens-style data, so it was included as a base learner initially. Out-of-fold evaluation put it at 1.0755 RMSE — worse than the plain Baseline model (0.9464), and the weakest of everything tested. It was also the slowest model in the set by a wide margin. Rather than let it drag the ensemble down while costing the most training time, it was cut. The final base model set is the five listed above.

### Hybridization

The content-based signal comes from TMDB synopses, embedded with `sentence-transformers` (`all-MiniLM-L6-v2`) and indexed with FAISS (`IndexFlatIP`, cosine similarity via normalized vectors). This powers two things:

- **"More like this (by plot)"** — direct nearest-neighbor lookup, genuinely content-based, distinct from the collaborative KNN engine (which finds movies rated similarly by the same users, regardless of what either movie is actually about)
- **Hybrid blend** — a per-user "taste vector" (average embedding of their highly-rated movies) blended with the ensemble's collaborative score at 85/15 weight, available as a toggle in the Advanced panel

This is a **post-hoc blend at inference time**, not a feature threaded into the meta-learner's leakage-safe OOF training. Properly integrating content similarity into the stacking pipeline would mean recomputing it per-fold with the same rigor as the statistical features — a larger rebuild than was justified for what's ultimately a secondary signal here. What's implemented is real (genuine embeddings, genuine vector search, genuinely blended into the final score), just architecturally simpler than a fully joint-trained system.

During development, the plot-similarity feature was found to return weak, coincidental matches once genuinely similar movies ran out — investigated, root-caused, and mitigated with a measured similarity threshold. Full writeup in [`ISSUES.md`](ISSUES.md).

---

## Project Structure

```
splice-engine/
├── .streamlit/
│   └── config.toml            # theme, disables the telemetry prompt
├── analysis/
│   ├── feedback_analysis.py   # calibration, engine comparison, rating trends
│   ├── generate_demo_data.py  # synthetic feedback for demos, kept separate from real data
│   ├── inspect_embeddings.py  # diagnostic: prints synopsis text + similarity scores
│   └── figures/
├── dashboard/
│   └── app.py                 # Streamlit app
├── data/
│   ├── movie_info.tsv         # MovieLens 100k
│   ├── user_info.tsv
│   ├── ratings.tsv
│   ├── ratings_train.tsv
│   ├── ratings_test.tsv
│   └── ml-1m/                 # MovieLens 1M (.dat files)
├── feedback/
│   └── ratings_log.csv        # grows as the dashboard is used
├── models/
│   ├── ml-100k/                # trained artifacts, per dataset
│   └── ml-1m/
├── notebooks/
│   └── Recommender.ipynb      # guided walkthrough, loads pre-trained models
├── src/
│   ├── data.py                 # ML-100k loader
│   ├── data_1m.py               # ML-1M loader
│   ├── features.py             # statistical features (shared)
│   ├── engines.py              # the three-tier router + ensemble (shared)
│   ├── embeddings.py           # content embeddings + FAISS index
│   ├── hybrid.py                # collaborative/content blending
│   ├── evaluate.py             # RMSE, precision@K, recall@K, coverage
│   ├── train.py                 # end-to-end training script
│   ├── tmdb.py                  # TMDB API wrapper, disk-cached
│   └── config.py                # .env / secrets loading
├── tests/
│   └── test_engines.py
├── .env.example
├── .gitignore
├── COLD_START.md
├── ISSUES.md
├── LICENSE
├── requirements.txt
├── run_dashboard.bat
└── README.md
```

---

## Setup

### 1. Environment

```bash
conda create -n myenv python=3.11
conda activate myenv
conda install pip
python -m pip install -r requirements.txt
```

`scikit-surprise` compiles a C extension on install — on Windows this needs a working C++ build toolchain. If it fails, installing via `pip` (rather than `conda`) generally resolves it, which is already how `requirements.txt` is set up.

### 2. Datasets

**MovieLens 100k** — already included under `data/`.

**MovieLens 1M** (optional, for the larger benchmark) — download from GroupLens and extract into `data/ml-1m/`:

```
data/ml-1m/users.dat
data/ml-1m/movies.dat
data/ml-1m/ratings.dat
```

Source: [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)

### 3. TMDB API key (for posters, synopses, and content embeddings)

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/)
2. Go to *Settings → API* and request a key (Developer, free, near-instant approval)
3. Copy the template and fill in your key:
   ```bash
   cp .env.example .env
   ```
   ```
   TMDB_API_KEY=your_key_here
   ```

> **`.env` is listed in `.gitignore` and will never be committed.**

### 4. Train the models

```bash
python -m src.train --dataset ml-100k
python -m src.train --dataset ml-1m      # optional, ~10x the data, proportionally longer
```

This runs the full pipeline — data loading, popularity/KNN engine construction, 10-fold out-of-fold base-model training, meta-learner training, full retrain, held-out evaluation — and saves everything to `models/<dataset>/`.

### 5. (Optional) Build the content index

Needed for "More like this (by plot)" and hybrid mode. Requires the TMDB key from step 3.

```bash
python -m src.embeddings --dataset ml-100k
```

Downloads the `all-MiniLM-L6-v2` model (~90MB) on first run and fetches a TMDB synopsis for every movie in the catalog. Both are cached — this only needs to run once per dataset.

### 6. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Or on Windows, double-click `run_dashboard.bat`.

### 7. (Optional) Run the test suite

```bash
python -m pytest tests/ -v
```

---

## Engineering Decisions

A running log of the judgment calls behind this project, kept honest rather than polished after the fact:

- **SVD++ was dropped** based on out-of-fold evidence (1.0755 RMSE, worse than the naive Baseline), not assumption. See [How It Works](#the-stacked-ensemble).
- **Testing caught two real bugs**, not hypothetical ones. `tests/test_engines.py` revealed that `get_knn_recs` could crash on a small catalog (missing a `min()` clamp on neighbor count), and — more significantly — that the ensemble's candidate generation only ever considered movies present in `train_master`, meaning any movie with zero ratings from anyone could never be recommended regardless of predicted quality. Both fixed; RMSE was unaffected since evaluation only scores existing test ratings, confirming the bug was silent rather than accuracy-affecting.
- **Content-embedding matching quality was investigated, not assumed.** A "More like this" result returning weak matches was traced to actual cosine similarity scores rather than dismissed or hidden. Full writeup in [`ISSUES.md`](ISSUES.md).
- **Hybridization is a post-hoc blend, not a joint-trained feature** — an explicit scope decision to avoid rebuilding the leakage-safe OOF pipeline for a secondary signal. Documented rather than glossed over.
- **Two datasets, trained and saved independently** — MovieLens 100k for fast iteration and the live dashboard demo, 1M as a "does this scale" benchmark. RMSE improved from 0.9010 to 0.8492 moving to 1M, consistent with more collaborative signal per user/item, though the two datasets use different train/test split methodology (100k uses GroupLens's own split; 1M uses a random split), so this isn't presented as a rigorously controlled scaling experiment.
- **Synthetic demo data is clearly separated from real usage data** — `feedback/demo_ratings_log.csv` vs. `feedback/ratings_log.csv`, never merged, generated data always labeled as such.
- **Hyperparameters are currently untuned** — carried over from initial development, not searched. A tuning pass is planned as a distinct final step, after the feature set was locked, to avoid tuning something that later changed.

---

## Acknowledgements

- [MovieLens datasets](https://grouplens.org/datasets/movielens/) — GroupLens Research, University of Minnesota. F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)
- [TMDB](https://www.themoviedb.org/) — this product uses the TMDB API but is not endorsed or certified by TMDB
- [scikit-surprise](https://surpriselib.com/) — Nicolas Hug
- [LightGBM](https://lightgbm.readthedocs.io/) — Microsoft
- [sentence-transformers](https://www.sbert.net/) — UKP Lab
- [FAISS](https://github.com/facebookresearch/faiss) — Meta AI Research
- [Streamlit](https://streamlit.io/)

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.
