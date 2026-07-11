"""
Generates synthetic feedback data so the analysis plots have enough volume
to look like something, without needing to sit and click through the
dashboard 100 times by hand.

This is explicitly NOT real usage data - it writes to a separate file
(feedback/demo_ratings_log.csv) rather than touching the real
feedback/ratings_log.csv, and analysis/feedback_analysis.py takes a --file
flag so you choose which one to read. Don't present demo output as if it
came from real users - the honest framing for a README/portfolio is
"simulated feedback used to demonstrate the analysis pipeline."

Ratings aren't pure noise - actual user_rating is generated as a function
of the real predicted score / real average rating plus randomness, so the
calibration plot shows a believable (imperfect) correlation rather than a
random scatter that wouldn't demonstrate anything.

Run with:
    python -m scripts.generate_demo_data
"""

import csv
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path

from src.data import load_and_clean
from src.engines import (
    build_ensemble_predict,
    get_profile_recs,
    score_by_genre_match,
    GENRE_PROFILES,
)

MODELS_DIR = Path("models/ml-100k")
OUTPUT_PATH = Path("feedback/demo_ratings_log.csv")

random.seed(7)


def load_models():
    with open(MODELS_DIR / "base_models.pkl", "rb") as f:
        base_models = pickle.load(f)
    with open(MODELS_DIR / "meta_learner.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(MODELS_DIR / "stat_features.pkl", "rb") as f:
        stat_features = pickle.load(f)

    ensemble_predict = build_ensemble_predict(
        base_models, meta, stat_features["user_stats"], stat_features["item_stats"], stat_features["global_mean"]
    )
    return ensemble_predict


def noisy_rating_from_prediction(predicted_1_to_5, noise_scale=1.2):
    """Predicted rating (1-5) -> a plausible user rating (0-10), with
    Gaussian noise so it's not a suspiciously perfect diagonal line."""
    base = predicted_1_to_5 * 2
    noisy = base + random.gauss(0, noise_scale)
    return int(round(max(0, min(10, noisy))))


def spread_timestamp(base_time, day_offset, jitter_minutes=180):
    return base_time + timedelta(days=day_offset, minutes=random.randint(-jitter_minutes, jitter_minutes))


def main():
    print("Loading data and models...")
    data = load_and_clean()
    train_master = data["train_master"]
    movie_info = data["movie_info"]
    user_info = data["user_info"]
    genre_cols = data["genre_cols"]
    ensemble_predict = load_models()

    rows = []
    base_time = datetime.now() - timedelta(days=14)

    # --- ensemble ratings: pick a handful of established users, generate
    # real predictions, derive a correlated (not identical) user rating ---
    established_users = (
        train_master.groupby("user_id").size().loc[lambda s: s >= 5].sample(6, random_state=7).index
    )
    for day, user_id in enumerate(established_users):
        seen = set(train_master.loc[train_master["user_id"] == user_id, "movie_id"])
        candidates = [m for m in movie_info["movie_id"].unique() if m not in seen][:40]
        sampled = random.sample(candidates, k=min(4, len(candidates)))

        for movie_id in sampled:
            title = movie_info.loc[movie_info["movie_id"] == movie_id, "title"].values[0]
            predicted = ensemble_predict(user_id, movie_id)
            rating = noisy_rating_from_prediction(predicted)
            rows.append({
                "timestamp": spread_timestamp(base_time, day).isoformat(timespec="seconds"),
                "movie_id": movie_id, "title": title, "engine_used": "ensemble",
                "predicted_score": round(predicted, 4), "user_rating": rating,
            })

    # --- profile ratings: no predicted score (genre-matched, not
    # rating-predicted), rating correlated with the movie's real average ---
    movie_avg = train_master.groupby("movie_id")["rating"].mean()
    for day, profile_name in enumerate(list(GENRE_PROFILES)[:6]):
        recs = get_profile_recs(profile_name, movie_info, genre_cols, train_master, n=4)
        for _, row in recs.iterrows():
            avg = movie_avg.get(row["movie_id"], 3.5)
            rating = noisy_rating_from_prediction(avg, noise_scale=1.5)
            rows.append({
                "timestamp": spread_timestamp(base_time, day + 2).isoformat(timespec="seconds"),
                "movie_id": int(row["movie_id"]), "title": row["title"],
                "engine_used": f"profile:{profile_name}", "predicted_score": "", "user_rating": rating,
            })

    # --- quiz ratings + an experience-rating trend that improves slightly
    # over time, a believable "we iterated on it and it got better" arc ---
    quiz_weight_sets = [
        {"genre_action": 1.0, "genre_adventure": 0.8},
        {"genre_comedy": 1.0},
        {"genre_drama": 1.0, "genre_romance": 0.6},
        {"genre_horror": 0.9, "genre_thriller": 0.7},
        {"genre_sci_fi": 1.0, "genre_fantasy": 0.6},
    ]
    for day, weights in enumerate(quiz_weight_sets):
        recs = score_by_genre_match(weights, movie_info, genre_cols, train_master, n=4)
        for _, row in recs.iterrows():
            avg = movie_avg.get(row["movie_id"], 3.5)
            rating = noisy_rating_from_prediction(avg, noise_scale=1.5)
            rows.append({
                "timestamp": spread_timestamp(base_time, day + 4).isoformat(timespec="seconds"),
                "movie_id": int(row["movie_id"]), "title": row["title"],
                "engine_used": "quiz", "predicted_score": "", "user_rating": rating,
            })

        # experience score trending gently upward with noise, days 4-8
        experience = min(10, max(0, round(6 + day * 0.7 + random.gauss(0, 0.8))))
        rows.append({
            "timestamp": spread_timestamp(base_time, day + 4, jitter_minutes=30).isoformat(timespec="seconds"),
            "movie_id": "", "title": "", "engine_used": "quiz_experience",
            "predicted_score": "", "user_rating": experience,
        })

    rows.sort(key=lambda r: r["timestamp"])

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "movie_id", "title", "engine_used", "predicted_score", "user_rating"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic feedback rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
