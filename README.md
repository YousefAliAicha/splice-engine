# Splice

A movie recommender system that adapts its strategy to how much is known about a user — from cold-start popularity rankings to a stacked ensemble of seven collaborative-filtering models — wrapped in an interactive dashboard that collects real user feedback to validate its own predictions.

Built on the MovieLens-100k dataset (943 users, ~1,680 movies, 100k ratings).

## How it works

| Ratings known | Engine                                                                                        |
| ------------- | --------------------------------------------------------------------------------------------- |
| 0             | Popularity ranking (gender-filtered)                                                          |
| 1–4           | Item-based KNN (cosine similarity)                                                            |
| 5+            | Stacked ensemble (SVD++, SVD, KNN, SlopeOne, Co-Clustering, Baseline → LightGBM meta-learner) |

## Status

🚧 Work in progress — refactoring from an exploratory notebook into a full package with a Streamlit dashboard and feedback-driven analysis.

## Setup

```bash
pip install -r requirements.txt
```

## License

MIT
