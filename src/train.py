"""
End-to-end training script. Run this to go from raw data to a saved,
ready-to-serve model.

    python -m src.train                  # MovieLens 100k (default)
    python -m src.train --dataset ml-1m  # MovieLens 1M

Saves the trained base models, meta-learner, popularity lists, and KNN
index to models/, so the dashboard and notebook don't need to retrain
from scratch every time.

1M is ~10x the ratings of 100k - expect the 10-fold OOF training step to
take proportionally longer. Nothing else in the pipeline changes; features.py
and engines.py don't know or care which dataset produced train_master.
"""

import argparse
import pickle
from pathlib import Path

from src.features import build_stat_features
from src.engines import (
    build_popularity,
    build_knn,
    get_base_models,
    train_oof_predictions,
    report_base_model_rmse,
    train_meta_learner,
    retrain_base_models_full,
    build_ensemble_predict,
)
from src.evaluate import evaluate_on_test


MODELS_DIR = Path("models")


def load_dataset(name):
    if name == "ml-1m":
        from src.data_1m import load_and_clean
        return load_and_clean()
    from src.data import load_and_clean
    return load_and_clean()


def main(dataset="ml-100k"):
    print(f"Loading and cleaning data ({dataset})...")
    data = load_dataset(dataset)
    output_dir = MODELS_DIR / dataset
    train_master = data["train_master"]
    test_master = data["test_master"]
    movie_info = data["movie_info"]

    print("\nBuilding popularity engine...")
    popularity_lists = build_popularity(train_master, movie_info)

    print("Building KNN engine...")
    knn = build_knn(train_master)

    print("\nTraining base models (10-fold OOF)...")
    base_models = get_base_models()
    oof_df = train_oof_predictions(train_master, base_models)

    print("\nBase model RMSEs:")
    base_rmses = report_base_model_rmse(oof_df, base_models)

    print("\nTraining meta-learner...")
    meta, eval_history = train_meta_learner(oof_df, base_models)

    print("\nRetraining base models on full training data...")
    base_models = retrain_base_models_full(train_master, base_models)

    stat_features = build_stat_features(train_master)
    ensemble_predict = build_ensemble_predict(
        base_models, meta, stat_features["user_stats"], stat_features["item_stats"], stat_features["global_mean"]
    )

    print("\nEvaluating on held-out test set...")
    test_rmse = evaluate_on_test(test_master, train_master, ensemble_predict)

    print("\nSaving models...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "base_models.pkl", "wb") as f:
        pickle.dump(base_models, f)

    with open(output_dir / "meta_learner.pkl", "wb") as f:
        pickle.dump(meta, f)

    with open(output_dir / "popularity_lists.pkl", "wb") as f:
        pickle.dump(popularity_lists, f)

    with open(output_dir / "knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open(output_dir / "stat_features.pkl", "wb") as f:
        pickle.dump(stat_features, f)

    with open(output_dir / "eval_history.pkl", "wb") as f:
        pickle.dump(eval_history, f)

    feature_cols = list(base_models.keys()) + ["user_bias", "user_std", "user_count", "item_bias", "item_std", "item_count"]
    importances = dict(zip(feature_cols, meta.feature_importances_.tolist()))
    with open(output_dir / "feature_importances.pkl", "wb") as f:
        pickle.dump(importances, f)

    # plain-text record of the run's numbers - the terminal output gets
    # buried under LightGBM/sklearn warnings, this doesn't
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"Dataset: {dataset}\n\n")
        f.write("Base model RMSEs (out-of-fold):\n")
        for name, score in base_rmses.items():
            f.write(f"  {name:12s} {score:.4f}\n")
        f.write(f"\nTest RMSE: {test_rmse:.4f}\n")

    print(f"Done. Models saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-100k")
    args = parser.parse_args()
    main(dataset=args.dataset)
