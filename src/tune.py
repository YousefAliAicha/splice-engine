"""
Hyperparameter tuning pass for the base models and meta-learner.

Deliberately NOT a full grid search or a full 10-fold OOF evaluation per
candidate - that would take hours for what's ultimately a secondary
improvement over an already-working system. Instead:

  1. Each base model gets a small random search (a handful of draws from
     a sensible range, not the whole space) evaluated on a single
     train/validation split - fast enough to try many candidates.
  2. Winning hyperparameters get plugged into get_base_models().
  3. One final full 10-fold OOF run (the real training procedure) confirms
     the tuned models actually still work correctly together and reports
     the real RMSE - the number that goes in the README, not the
     single-split search numbers.

Run with:
    python -m src.tune --dataset ml-100k --iterations 15
"""

import argparse
import random
from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as sk_train_test_split
from surprise import SVD, KNNWithMeans, BaselineOnly, SlopeOne, CoClustering
from surprise import Dataset as SurpriseDataset, Reader
import lightgbm as lgb
import pandas as pd

random.seed(42)


SEARCH_SPACES = {
    "SVD": {
        "n_factors": [50, 100, 150, 200],
        "n_epochs": [20, 40, 60],
        "lr_all": [0.002, 0.005, 0.008],
        "reg_all": [0.01, 0.02, 0.05],
    },
    "iKNN": {
        "k": [20, 30, 40, 60],
        "min_k": [1, 3, 5],
    },
    "SlopeOne": {},  # no meaningful hyperparameters to search
    "Baseline": {},  # no meaningful hyperparameters to search
    "CoClustering": {
        "n_cltr_u": [3, 5, 8],
        "n_cltr_i": [3, 5, 8],
        "n_epochs": [15, 20, 30],
    },
}


def sample_params(space):
    return {k: random.choice(v) for k, v in space.items()}


def build_model(name, params):
    if name == "SVD":
        return SVD(random_state=42, **params)
    if name == "iKNN":
        return KNNWithMeans(sim_options={"name": "pearson_baseline", "user_based": False}, verbose=False, **params)
    if name == "SlopeOne":
        return SlopeOne()
    if name == "Baseline":
        return BaselineOnly(verbose=False)
    if name == "CoClustering":
        return CoClustering(random_state=42, **params)
    raise ValueError(f"Unknown model: {name}")


def tune_one_model(name, trainset, validset, iterations):
    space = SEARCH_SPACES[name]

    if not space:
        # nothing to search - just fit once and report baseline RMSE
        model = build_model(name, {})
        model.fit(trainset)
        preds = model.test(validset)
        rmse = sqrt(mean_squared_error([p.r_ui for p in preds], [p.est for p in preds]))
        print(f"  {name:12s} (no search space)  RMSE: {rmse:.4f}")
        return {}, rmse

    best_params, best_rmse = None, float("inf")
    for i in range(iterations):
        params = sample_params(space)
        model = build_model(name, params)
        model.fit(trainset)
        preds = model.test(validset)
        rmse = sqrt(mean_squared_error([p.r_ui for p in preds], [p.est for p in preds]))

        marker = ""
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params
            marker = "  <- best so far"
        print(f"  {name:12s} [{i+1}/{iterations}] {params}  RMSE: {rmse:.4f}{marker}")

    print(f"  {name:12s} BEST: {best_params}  RMSE: {best_rmse:.4f}")
    return best_params, best_rmse


def tune_meta_learner(X_train, y_train, X_valid, y_valid, iterations):
    space = {
        "n_estimators": [200, 300, 500, 700],
        "learning_rate": [0.02, 0.05, 0.08],
        "max_depth": [3, 5, 7],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 30],
        "reg_alpha": [0.0, 0.1, 0.3],
        "reg_lambda": [0.0, 0.1, 0.3],
    }

    best_params, best_rmse = None, float("inf")
    for i in range(iterations):
        params = sample_params(space)
        model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, subsample=0.8, colsample_bytree=0.8, **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        rmse = sqrt(mean_squared_error(y_valid, preds))

        marker = ""
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params
            marker = "  <- best so far"
        print(f"  meta-learner [{i+1}/{iterations}] {params}  RMSE: {rmse:.4f}{marker}")

    print(f"  meta-learner BEST: {best_params}  RMSE: {best_rmse:.4f}")
    return best_params, best_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-100k")
    parser.add_argument("--iterations", type=int, default=15, help="Random search draws per model")
    args = parser.parse_args()

    if args.dataset == "ml-1m":
        from src.data_1m import load_and_clean
    else:
        from src.data import load_and_clean

    print(f"Loading {args.dataset}...")
    data = load_and_clean()
    train_master = data["train_master"]

    # single train/validation split for the search itself - NOT the same
    # as the 10-fold OOF procedure used for real training. This is a
    # cheap proxy to compare candidates quickly; the winning config gets
    # a proper OOF run afterward for the real reported number.
    surprise_data = SurpriseDataset.load_from_df(
        train_master[["user_id", "movie_id", "rating"]], Reader(rating_scale=(1, 5))
    )
    raw_train, raw_valid = sk_train_test_split(surprise_data.raw_ratings, test_size=0.2, random_state=42)
    surprise_data.raw_ratings = raw_train
    trainset = surprise_data.build_full_trainset()
    validset = [(u, i, r) for (u, i, r, _) in raw_valid]

    print(f"\n=== Tuning base models ({args.iterations} draws each, single-split validation) ===\n")
    best_configs = {}
    for name in SEARCH_SPACES:
        best_params, _ = tune_one_model(name, trainset, validset, args.iterations)
        best_configs[name] = best_params
        print()

    print("=== Best base model configs found ===")
    for name, params in best_configs.items():
        print(f"  {name}: {params}")

    # Build OOF-style features on the same split for meta-learner tuning -
    # simplified (not the full 10-fold procedure) since this is just for
    # picking meta-learner hyperparameters, not the final reported model.
    print("\n=== Preparing features for meta-learner tuning ===")
    tuned_models = {name: build_model(name, params) for name, params in best_configs.items()}
    for model in tuned_models.values():
        model.fit(trainset)

    global_mean = train_master["rating"].mean()
    from src.features import compute_user_stats, compute_item_stats

    valid_df = pd.DataFrame(validset, columns=["user_id", "movie_id", "rating"])
    user_stats = compute_user_stats(train_master, global_mean)
    item_stats = compute_item_stats(train_master, global_mean)

    from src.features import get_stat_features
    feature_cols = list(tuned_models.keys()) + ["user_bias", "user_std", "user_count", "item_bias", "item_std", "item_count"]
    rows = []
    for row in valid_df.itertuples():
        base_preds = [tuned_models[name].predict(row.user_id, row.movie_id).est for name in tuned_models]
        sf = get_stat_features(row.user_id, row.movie_id, user_stats, item_stats, global_mean)
        rows.append(base_preds + [sf[c] for c in feature_cols[len(tuned_models):]] + [row.rating])
    feat_df = pd.DataFrame(rows, columns=feature_cols + ["true"])

    # split this feature set again for the meta-learner's own train/valid
    meta_train, meta_valid = sk_train_test_split(feat_df, test_size=0.3, random_state=42)

    print(f"\n=== Tuning meta-learner ({args.iterations} draws, single-split validation) ===\n")
    best_meta_params, _ = tune_meta_learner(
        meta_train[feature_cols], meta_train["true"], meta_valid[feature_cols], meta_valid["true"], args.iterations
    )

    print("\n=== DONE ===")
    print("Best base model configs:")
    for name, params in best_configs.items():
        print(f"  {name}: {params}")
    print(f"Best meta-learner config: {best_meta_params}")
    print(
        "\nThese were found on a single train/validation split for speed. "
        "Update get_base_models() and train_meta_learner() in src/engines.py "
        "with these values, then rerun `python -m src.train` for the real "
        "10-fold OOF confirmation number."
    )


if __name__ == "__main__":
    main()
