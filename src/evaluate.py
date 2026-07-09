"""
Evaluation metrics. RMSE is the headline number, but for a system that's
really about ranked recommendations, precision/recall @K and coverage tell
you more about whether the top-N list is actually useful.
"""

from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def evaluate_on_test(test_master, train_master, ensemble_predict):
    """Held-out RMSE. Only run this once, at the very end - this is the
    only legitimate use of test_master.

    Cold-start rows (user or movie never seen in training) get skipped
    since the ensemble has no way to score them - that's what the KNN/
    popularity engines are for, and they're not what we're grading here.
    """
    trained_users = set(train_master["user_id"].unique())
    trained_movies = set(train_master["movie_id"].unique())

    known = test_master[
        test_master["user_id"].isin(trained_users) & test_master["movie_id"].isin(trained_movies)
    ]

    preds = [ensemble_predict(row.user_id, row.movie_id) for row in known.itertuples()]
    preds = np.clip(preds, 1, 5)

    score = rmse(known["rating"].values, preds)

    print(f"Total test ratings:  {len(test_master):,}")
    print(f"Known u+m pairs:     {len(known):,}")
    print(f"Cold-start (skipped): {len(test_master) - len(known):,}")
    print(f"Test RMSE: {score:.4f}")

    return score


def precision_recall_at_k(test_master, train_master, ensemble_predict, k=10, relevance_threshold=4):
    """For each test user, take the top-K predicted movies and check how
    many were actually rated >= relevance_threshold by that user in the
    test set. Averaged across users.

    This is a rougher metric than RMSE - it only considers movies the user
    actually rated in the test set as candidates, not the full catalog -
    but it's a decent proxy for "did the ranked list look good" without
    needing to score every movie for every user.
    """
    trained_users = set(train_master["user_id"].unique())
    trained_movies = set(train_master["movie_id"].unique())

    precisions, recalls = [], []

    for user_id, group in test_master.groupby("user_id"):
        if user_id not in trained_users:
            continue

        candidates = group[group["movie_id"].isin(trained_movies)]
        if candidates.empty:
            continue

        scored = [
            (row.movie_id, ensemble_predict(user_id, row.movie_id), row.rating)
            for row in candidates.itertuples()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        relevant_in_top_k = sum(1 for _, _, true_rating in top_k if true_rating >= relevance_threshold)
        relevant_total = sum(1 for _, _, true_rating in scored if true_rating >= relevance_threshold)

        precisions.append(relevant_in_top_k / len(top_k))
        if relevant_total > 0:
            recalls.append(relevant_in_top_k / relevant_total)

    return {
        "precision_at_k": np.mean(precisions) if precisions else 0.0,
        "recall_at_k": np.mean(recalls) if recalls else 0.0,
        "k": k,
        "users_evaluated": len(precisions),
    }


def catalog_coverage(all_recommended_ids, movie_info):
    """What fraction of the whole catalog ever shows up as a recommendation.
    Low coverage means the system keeps recommending the same popular
    handful of movies regardless of who's asking."""
    total_movies = movie_info["movie_id"].nunique()
    unique_recommended = len(set(all_recommended_ids))
    return unique_recommended / total_movies
