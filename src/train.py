"""
End-to-end training script. Run this to go from raw TSVs to a saved,
ready-to-serve model.

    python -m src.train

Saves the trained base models, meta-learner, popularity lists, and KNN
index to models/, so the dashboard and notebook don't need to retrain
from scratch every time.
"""

import pickle
from pathlib import Path

from src.data import load_and_clean
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


def main():
    print("Loading and cleaning data...")
    data = load_and_clean()
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
    MODELS_DIR.mkdir(exist_ok=True)

    with open(MODELS_DIR / "base_models.pkl", "wb") as f:
        pickle.dump(base_models, f)

    with open(MODELS_DIR / "meta_learner.pkl", "wb") as f:
        pickle.dump(meta, f)

    with open(MODELS_DIR / "popularity_lists.pkl", "wb") as f:
        pickle.dump(popularity_lists, f)

    with open(MODELS_DIR / "knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open(MODELS_DIR / "stat_features.pkl", "wb") as f:
        pickle.dump(stat_features, f)

    with open(MODELS_DIR / "eval_history.pkl", "wb") as f:
        pickle.dump(eval_history, f)

    # plain-text record of the run's numbers - the terminal output gets
    # buried under LightGBM/sklearn warnings, this doesn't
    with open(MODELS_DIR / "results.txt", "w") as f:
        f.write("Base model RMSEs (out-of-fold):\n")
        for name, score in base_rmses.items():
            f.write(f"  {name:12s} {score:.4f}\n")
        f.write(f"\nTest RMSE: {test_rmse:.4f}\n")

    print(f"Done. Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
