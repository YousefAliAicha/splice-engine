"""
Statistical features for the meta-learner: per-user and per-item bias,
spread, and rating count. These get computed once from train_master and
looked up per (user, movie) pair when building the feature matrix.

There's also a couple of helpers at the bottom for pulling genre vectors
out of movie_info - not used yet, but useful once we want to add
content-based signal into the ensemble.
"""

import pandas as pd


def compute_global_mean(train_master):
    return train_master["rating"].mean()


def compute_user_stats(train_master, global_mean=None):
    if global_mean is None:
        global_mean = compute_global_mean(train_master)

    stats = train_master.groupby("user_id")["rating"].agg(["mean", "std", "count"])
    stats = stats.fillna(0)  # users with only 1 rating have no std - treat as 0, not NaN
    stats["bias"] = stats["mean"] - global_mean
    return stats


def compute_item_stats(train_master, global_mean=None):
    if global_mean is None:
        global_mean = compute_global_mean(train_master)

    stats = train_master.groupby("movie_id")["rating"].agg(["mean", "std", "count"])
    stats = stats.fillna(0)
    stats["bias"] = stats["mean"] - global_mean
    return stats


def build_stat_features(train_master):
    """One-shot helper - computes global mean plus user/item stats and
    bundles them together so you don't have to pass three things around
    separately."""
    global_mean = compute_global_mean(train_master)
    return {
        "global_mean": global_mean,
        "user_stats": compute_user_stats(train_master, global_mean),
        "item_stats": compute_item_stats(train_master, global_mean),
    }


def get_stat_features(user_id, movie_id, user_stats, item_stats, global_mean):
    """Look up the six stat features for one (user, movie) pair.

    Users/movies with no training history (cold start) get neutral
    defaults - zero bias, zero std, zero count - rather than blowing up
    on a missing key."""
    if user_id in user_stats.index:
        u = user_stats.loc[user_id]
        user_bias, user_std, user_count = u["bias"], u["std"], u["count"]
    else:
        user_bias, user_std, user_count = 0.0, 0.0, 0

    if movie_id in item_stats.index:
        i = item_stats.loc[movie_id]
        item_bias, item_std, item_count = i["bias"], i["std"], i["count"]
    else:
        item_bias, item_std, item_count = 0.0, 0.0, 0

    return {
        "user_bias": user_bias,
        "user_std": user_std,
        "user_count": user_count,
        "item_bias": item_bias,
        "item_std": item_std,
        "item_count": item_count,
    }


# --- content features (genres) - not wired into the ensemble yet ---

def build_genre_matrix(movie_info, genre_cols):
    """movie_id -> binary genre columns, indexed for quick lookup."""
    return movie_info.set_index("movie_id")[genre_cols]


def get_genre_vector(movie_id, genre_matrix):
    if movie_id not in genre_matrix.index:
        return None
    return genre_matrix.loc[movie_id]
