"""
The three recommendation engines, picked based on how much history a user
has:

    0 ratings   -> popularity engine (just rank by average, filtered by gender)
    1-4 ratings -> item-based KNN (not enough data for the ensemble to be reliable)
    5+ ratings  -> full stacked ensemble (seven CF models + a GBM meta-learner)

The router at the bottom (`recommend`) is the single entry point everything
else calls into.
"""

from math import sqrt

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, KNNWithMeans, BaselineOnly, SlopeOne, CoClustering
from surprise import Dataset as SurpriseDataset, Reader
from surprise.model_selection import KFold

from src.features import get_stat_features


# ---------------------------------------------------------------------------
# Popularity engine - the cold-start fallback
# ---------------------------------------------------------------------------

def build_popularity(train_master, movie_info, min_ratings=30):
    """Top-10 movies by average rating, split overall / male / female.

    min_ratings filters out movies that only have a handful of ratings -
    otherwise some obscure film with two 5-star ratings tops the chart,
    which isn't a useful recommendation for anyone.
    """
    titles = movie_info[["movie_id", "title"]].drop_duplicates()

    def top_n(subset):
        agg = subset.groupby("movie_id")["rating"].agg(["count", "mean"])
        agg = agg[agg["count"] >= min_ratings]
        agg = agg.reset_index().merge(titles, on="movie_id")
        return agg.sort_values("mean", ascending=False).head(10)[["title", "mean", "count"]]

    return {
        "overall": top_n(train_master),
        "male": top_n(train_master[train_master["gender"] == "M"]),
        "female": top_n(train_master[train_master["gender"] == "F"]),
    }


def score_by_genre_match(genre_weights, movie_info, genre_cols, train_master, min_ratings=20, n=10):
    """Rank the catalog against a genre-affinity vector (e.g. {'genre_action': 1,
    'genre_comedy': 0.5}), weighted by how well-liked each movie actually is.

    Used for the curated "if you like X" profiles and the quiz results - neither
    of those need a specific user_id, just a taste direction.
    """
    stats = train_master.groupby("movie_id")["rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_ratings]

    candidates = movie_info[movie_info["movie_id"].isin(stats.index)].copy()
    candidates = candidates.set_index("movie_id")

    genre_match = sum(candidates[g] * w for g, w in genre_weights.items() if w > 0)
    avg_rating = stats.loc[candidates.index, "mean"]

    # weighted rating carries more of the ranking than genre match alone -
    # otherwise a mediocre movie that happens to tick every genre box beats
    # a genuinely great one that only ticks one
    combined = genre_match * 0.4 + avg_rating * 0.6

    ranked = candidates.assign(score=combined).sort_values("score", ascending=False)
    return ranked.head(n)[["title"]].reset_index()


GENRE_PROFILES = {
    "Action & Adventure": {"genre_action": 1.0, "genre_adventure": 1.0, "genre_war": 0.4},
    "Romance": {"genre_romance": 1.0, "genre_drama": 0.3},
    "Comedy": {"genre_comedy": 1.0},
    "Sci-Fi & Fantasy": {"genre_sci_fi": 1.0, "genre_fantasy": 0.8},
    "Horror & Thriller": {"genre_horror": 1.0, "genre_thriller": 0.8},
    "Drama": {"genre_drama": 1.0},
    "Family & Animation": {"genre_animation": 1.0, "genre_childrens": 1.0},
    "Crime & Mystery": {"genre_crime": 1.0, "genre_mystery": 0.8, "genre_film_noir": 0.5},
    "Musicals": {"genre_musical": 1.0},
    "Classics": {"genre_drama": 0.4, "genre_war": 0.3, "genre_western": 0.5},
}


def get_profile_recs(profile_name, movie_info, genre_cols, train_master, n=10):
    weights = GENRE_PROFILES[profile_name]
    return score_by_genre_match(weights, movie_info, genre_cols, train_master, n=n)


def get_popularity_recs(popularity_lists, gender=None, n=10):
    key = "male" if gender == "M" else "female" if gender == "F" else "overall"
    return popularity_lists[key].head(n)["title"].tolist()


# ---------------------------------------------------------------------------
# KNN engine - for users with a handful of ratings (1-4)
# ---------------------------------------------------------------------------

def build_knn(train_master, n_neighbors=20):
    """Item-based KNN over a movie x user rating matrix, cosine similarity.

    Missing ratings get filled with 0 rather than dropped - it's a crude
    approach (0 isn't really "no rating", it's a rating of zero) but it's
    what makes the matrix dense enough for brute-force cosine to be cheap.
    """
    pivot = train_master.pivot_table(index="movie_id", columns="user_id", values="rating").fillna(0)

    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(csr_matrix(pivot.values))

    return {"pivot": pivot, "model": model}


def get_knn_recs(movie_id, knn, movie_info, n=10):
    pivot, model = knn["pivot"], knn["model"]

    if movie_id not in pivot.index:
        return []

    row_idx = pivot.index.get_loc(movie_id)
    vec = pivot.iloc[row_idx].values.reshape(1, -1)

    neighbor_idxs = model.kneighbors(vec, n_neighbors=n + 1)[1].flatten()
    neighbor_idxs = neighbor_idxs[1:]  # first neighbor is always the movie itself

    neighbor_ids = pivot.index[neighbor_idxs]
    titles_lookup = movie_info.set_index("movie_id")["title"]
    return [titles_lookup.get(mid, "Unknown") for mid in neighbor_ids]


# ---------------------------------------------------------------------------
# Base models - seven collaborative filtering algorithms, blended via
# out-of-fold predictions so the meta-learner never sees a prediction from
# a model that was trained on that same row.
# ---------------------------------------------------------------------------

def get_base_models(random_state=42):
    # SVDpp was tested and dropped - OOF RMSE of 1.0755, worse than plain
    # Baseline (0.9464). Slowest model in the set by a wide margin too,
    # so cutting it also speeds up every retrain. See README for details.
    return {
        "SVD": SVD(n_factors=200, n_epochs=60, lr_all=0.005, reg_all=0.015, random_state=random_state),
        "iKNN": KNNWithMeans(k=40, sim_options={"name": "pearson_baseline", "user_based": False}, verbose=False),
        "Baseline": BaselineOnly(verbose=False),
        "SlopeOne": SlopeOne(),
        "CoClustering": CoClustering(random_state=random_state),
    }


def train_oof_predictions(train_master, base_models, n_splits=10, random_state=42):
    """10-fold CV to get out-of-fold predictions for every training row.

    Each row's prediction comes from a model that never saw that row during
    training - this is what keeps the meta-learner honest, otherwise it'd
    just learn to trust whichever base model overfit the hardest.
    """
    surprise_data = SurpriseDataset.load_from_df(
        train_master[["user_id", "movie_id", "rating"]], Reader(rating_scale=(1, 5))
    )
    kf = KFold(n_splits=n_splits, random_state=random_state)

    global_mean = train_master["rating"].mean()
    oof_rows = []

    for fold_idx, (fold_train, fold_test) in enumerate(kf.split(surprise_data)):
        # stats computed from THIS fold's training portion only
        fold_train_df = pd.DataFrame(fold_train.build_testset(), columns=["user_id", "movie_id", "rating"])
        user_stats = fold_train_df.groupby("user_id")["rating"].agg(["mean", "std", "count"]).fillna(0)
        item_stats = fold_train_df.groupby("movie_id")["rating"].agg(["mean", "std", "count"]).fillna(0)
        user_stats["bias"] = user_stats["mean"] - global_mean
        item_stats["bias"] = item_stats["mean"] - global_mean

        fold_preds = {}
        for name, model in base_models.items():
            model.fit(fold_train)
            fold_preds[name] = {(p.uid, p.iid): p.est for p in model.test(fold_test)}

        for uid, iid, true_rating in fold_test:
            row = {"user_id": uid, "movie_id": iid, "true": true_rating}
            for name in base_models:
                row[name] = fold_preds[name].get((uid, iid), global_mean)
            sf = get_stat_features(uid, iid, user_stats, item_stats, global_mean)
            row.update(sf)
            oof_rows.append(row)

    return pd.DataFrame(oof_rows)


def report_base_model_rmse(oof_df, base_models):
    """Print each base model's standalone RMSE - useful for spotting a
    model that's dragging the ensemble down (anything above ~0.97 on this
    dataset is a candidate to drop)."""
    scores = {}
    for name in base_models:
        rmse = sqrt(mean_squared_error(oof_df["true"], oof_df[name]))
        scores[name] = rmse
        print(f"{name:12s} RMSE: {rmse:.4f}")
    return scores


# ---------------------------------------------------------------------------
# Meta-learner - LightGBM, trained on the OOF predictions + stat features
# ---------------------------------------------------------------------------

FEATURE_COLS = ["user_bias", "user_std", "user_count", "item_bias", "item_std", "item_count"]


def train_meta_learner(oof_df, base_models):
    """Trains the LightGBM meta-learner on the OOF predictions.

    Splits off 15% of the OOF rows as an internal validation set purely so
    LightGBM can record a real train-vs-validation RMSE curve per boosting
    round (used for the learning-curve plot in the dashboard) - this is
    separate from, and has nothing to do with, test_master.
    """
    feature_cols = list(base_models.keys()) + FEATURE_COLS
    X = oof_df[feature_cols]
    y = oof_df["true"].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)

    meta = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    eval_history = {}
    meta.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_names=["train", "valid"],
        eval_metric="rmse",
        callbacks=[lgb.record_evaluation(eval_history)],
    )

    # refit on the full OOF set now that we've got the learning curve - no
    # reason to throw away 15% of the data for the model that actually ships
    meta.fit(X, y)

    oof_preds = np.clip(meta.predict(X), 1, 5)
    rmse = sqrt(mean_squared_error(y, oof_preds))
    print(f"Meta-learner OOF RMSE: {rmse:.4f}")

    return meta, eval_history


def retrain_base_models_full(train_master, base_models):
    """Base models only saw 90% of the data per fold during OOF training -
    once the meta-learner's weights are locked in, retrain everything on
    the full training set so inference gets the benefit of all the data."""
    full_data = SurpriseDataset.load_from_df(
        train_master[["user_id", "movie_id", "rating"]], Reader(rating_scale=(1, 5))
    )
    full_trainset = full_data.build_full_trainset()

    for name, model in base_models.items():
        model.fit(full_trainset)

    return base_models


def build_ensemble_predict(base_models, meta, user_stats, item_stats, global_mean):
    """Returns a predict(user_id, movie_id) function that runs a rating
    through every base model, gathers the stat features, and hands the lot
    to the meta-learner."""
    feature_cols = list(base_models.keys()) + FEATURE_COLS

    def predict(user_id, movie_id):
        base_preds = [base_models[name].predict(user_id, movie_id).est for name in base_models]
        sf = get_stat_features(user_id, movie_id, user_stats, item_stats, global_mean)
        row = base_preds + [sf[col] for col in FEATURE_COLS]
        X = pd.DataFrame([row], columns=feature_cols)
        raw = meta.predict(X)[0]
        return float(np.clip(raw, 1, 5))

    return predict


# ---------------------------------------------------------------------------
# Router - the single entry point downstream code (dashboard, notebook) uses
# ---------------------------------------------------------------------------

def recommend(user_id, train_master, movie_info, popularity_lists, knn, ensemble_predict, user_info=None, n=10):
    """Picks an engine based on how many ratings this user has and returns
    a list of (movie_id, title, predicted_score) tuples."""
    seen = set(train_master.loc[train_master["user_id"] == user_id, "movie_id"])
    rating_count = len(seen)

    if rating_count == 0:
        gender = None
        if user_info is not None:
            match = user_info.loc[user_info["user_id"] == user_id, "gender"]
            gender = match.values[0] if len(match) else None
        titles = get_popularity_recs(popularity_lists, gender=gender, n=n)
        return [(None, t, None) for t in titles]

    if rating_count < 5:
        liked = train_master[(train_master["user_id"] == user_id) & (train_master["rating"] >= 4)]
        if liked.empty:
            liked = train_master[train_master["user_id"] == user_id]
        seed_movie = liked.sort_values("rating", ascending=False).iloc[0]["movie_id"]
        titles = get_knn_recs(seed_movie, knn, movie_info, n=n)
        return [(None, t, None) for t in titles]

    # 5+ ratings -> full ensemble, scored across every unseen movie
    candidate_ids = [mid for mid in train_master["movie_id"].unique() if mid not in seen]
    titles_lookup = movie_info.set_index("movie_id")["title"]

    scored = [(mid, titles_lookup.get(mid, "Unknown"), ensemble_predict(user_id, mid)) for mid in candidate_ids]
    scored.sort(key=lambda row: row[2], reverse=True)
    return scored[:n]
