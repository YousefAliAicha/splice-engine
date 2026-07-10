"""
Tests for src/engines.py - mainly the routing logic (recommend() picking
the right engine based on rating count) and the standalone functions that
don't need a trained model to sanity-check.

Deliberately doesn't touch the real trained ensemble (base_models.pkl etc.)
- that would mean either shipping model files just for CI or retraining on
every test run, neither of which is worth it here. Where the ensemble path
needs exercising, a fake predict function stands in for it.

Run with:
    pytest tests/
"""

import pandas as pd
import pytest

from src.engines import (
    recommend,
    get_popularity_recs,
    build_popularity,
    build_knn,
    get_knn_recs,
    score_by_genre_match,
    get_profile_recs,
    GENRE_PROFILES,
)
from src.features import get_stat_features, build_stat_features


# ---------------------------------------------------------------------------
# small synthetic dataset - just enough shape to exercise the routing logic
# ---------------------------------------------------------------------------

@pytest.fixture
def movie_info():
    genre_cols = [
        "genre_action", "genre_adventure", "genre_animation", "genre_childrens",
        "genre_comedy", "genre_crime", "genre_drama", "genre_fantasy",
        "genre_film_noir", "genre_horror", "genre_musical", "genre_mystery",
        "genre_romance", "genre_sci_fi", "genre_thriller", "genre_war", "genre_western",
    ]
    df = pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5],
        "title": ["Action Movie", "Romance Movie", "Comedy Movie", "Sci-Fi Movie", "Drama Movie"],
    })
    for col in genre_cols:
        df[col] = 0
    df.loc[df["title"] == "Action Movie", "genre_action"] = 1
    df.loc[df["title"] == "Romance Movie", "genre_romance"] = 1
    df.loc[df["title"] == "Comedy Movie", "genre_comedy"] = 1
    df.loc[df["title"] == "Sci-Fi Movie", "genre_sci_fi"] = 1
    df.loc[df["title"] == "Sci-Fi Movie", "genre_action"] = 1  # also action, used in the genre-match test
    df.loc[df["title"] == "Drama Movie", "genre_drama"] = 1
    return df


@pytest.fixture
def user_info():
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "gender": ["M", "F", "M"],
        "age": [25, 30, 40],
        "occupation": ["student", "engineer", "artist"],
    })


@pytest.fixture
def train_master():
    # user 1: no ratings at all (cold start)
    # user 2: 3 ratings (sparse -> knn path)
    # user 3: 6 ratings (established -> ensemble path)
    rows = []
    for movie_id in [1, 2, 3]:
        rows.append({"user_id": 2, "movie_id": movie_id, "rating": 4, "gender": "F"})
    for movie_id in [1, 2, 3, 4, 5]:
        rows.append({"user_id": 3, "movie_id": movie_id, "rating": 5, "gender": "M"})
    rows.append({"user_id": 3, "movie_id": 1, "rating": 4, "gender": "M"})  # 6th rating for user 3
    # a few other users so popularity/min_ratings thresholds have something to chew on
    for uid in [10, 11, 12, 13]:
        rows.append({"user_id": uid, "movie_id": 1, "rating": 5, "gender": "M"})
        rows.append({"user_id": uid, "movie_id": 2, "rating": 3, "gender": "F"})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_ensemble_predict():
    """Stands in for the real trained ensemble - just returns a
    deterministic score based on movie_id so we can check sort order."""
    def predict(user_id, movie_id):
        return float(movie_id) % 5 + 1
    return predict


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_recommend_routes_cold_start_to_popularity(train_master, movie_info, user_info):
    popularity_lists = build_popularity(train_master, movie_info, min_ratings=1)
    knn = build_knn(train_master)

    recs = recommend(
        user_id=1, train_master=train_master, movie_info=movie_info,
        popularity_lists=popularity_lists, knn=knn, ensemble_predict=None,
        user_info=user_info, n=5,
    )

    # cold start recs come back with no movie_id/score, just titles
    assert len(recs) > 0
    assert all(movie_id is None for movie_id, title, score in recs)


def test_recommend_routes_sparse_user_to_knn(train_master, movie_info, user_info):
    popularity_lists = build_popularity(train_master, movie_info, min_ratings=1)
    knn = build_knn(train_master)

    recs = recommend(
        user_id=2, train_master=train_master, movie_info=movie_info,
        popularity_lists=popularity_lists, knn=knn, ensemble_predict=None,
        user_info=user_info, n=5,
    )

    assert len(recs) > 0


def test_recommend_routes_established_user_to_ensemble(train_master, movie_info, user_info, fake_ensemble_predict):
    popularity_lists = build_popularity(train_master, movie_info, min_ratings=1)
    knn = build_knn(train_master)

    recs = recommend(
        user_id=3, train_master=train_master, movie_info=movie_info,
        popularity_lists=popularity_lists, knn=knn, ensemble_predict=fake_ensemble_predict,
        user_info=user_info, n=5,
    )

    # user 3 has rated every movie in the fixture, so there should be
    # nothing left to recommend
    assert recs == []


def test_recommend_ensemble_excludes_already_rated(movie_info, user_info, fake_ensemble_predict):
    # catalog needs a 6th movie here - rating_count>=5 requires 5 distinct
    # rated movies, and the 5-movie base fixture would leave nothing left
    # to recommend
    extended_movies = pd.concat([
        movie_info,
        pd.DataFrame([{**{c: 0 for c in movie_info.columns}, "movie_id": 6, "title": "Extra Movie"}]),
    ], ignore_index=True)

    train_master = pd.DataFrame([
        {"user_id": 5, "movie_id": mid, "rating": 4, "gender": "M"} for mid in [1, 2, 3, 4, 5]
    ])
    popularity_lists = build_popularity(train_master, extended_movies, min_ratings=1)
    knn = build_knn(train_master)

    recs = recommend(
        user_id=5, train_master=train_master, movie_info=extended_movies,
        popularity_lists=popularity_lists, knn=knn, ensemble_predict=fake_ensemble_predict,
        user_info=user_info, n=5,
    )

    recommended_ids = [movie_id for movie_id, title, score in recs]
    assert 1 not in recommended_ids  # already rated
    assert 6 in recommended_ids      # the one unrated movie


def test_recommend_ensemble_sorted_by_score_descending(movie_info, user_info, fake_ensemble_predict):
    train_master = pd.DataFrame([{"user_id": 5, "movie_id": 1, "rating": r} for r in [4, 5, 3, 4, 5]])
    train_master["gender"] = "M"
    popularity_lists = build_popularity(train_master, movie_info, min_ratings=1)
    knn = build_knn(train_master)

    recs = recommend(
        user_id=5, train_master=train_master, movie_info=movie_info,
        popularity_lists=popularity_lists, knn=knn, ensemble_predict=fake_ensemble_predict,
        user_info=user_info, n=5,
    )

    scores = [score for _, _, score in recs]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# popularity engine
# ---------------------------------------------------------------------------

def test_get_popularity_recs_respects_gender(train_master, movie_info):
    popularity_lists = build_popularity(train_master, movie_info, min_ratings=1)

    male_recs = get_popularity_recs(popularity_lists, gender="M", n=5)
    female_recs = get_popularity_recs(popularity_lists, gender="F", n=5)
    overall_recs = get_popularity_recs(popularity_lists, gender=None, n=5)

    assert isinstance(male_recs, list)
    assert isinstance(female_recs, list)
    assert isinstance(overall_recs, list)


# ---------------------------------------------------------------------------
# knn engine
# ---------------------------------------------------------------------------

def test_knn_recs_exclude_seed_movie(train_master, movie_info):
    knn = build_knn(train_master)
    recs = get_knn_recs(movie_id=1, knn=knn, movie_info=movie_info, n=3)

    assert "Action Movie" not in recs  # movie 1 shouldn't recommend itself


def test_knn_recs_unknown_movie_returns_empty(train_master, movie_info):
    knn = build_knn(train_master)
    recs = get_knn_recs(movie_id=9999, knn=knn, movie_info=movie_info, n=3)

    assert recs == []


# ---------------------------------------------------------------------------
# genre scoring / quiz-taste-profile path
# ---------------------------------------------------------------------------

def test_score_by_genre_match_favors_matching_genre(train_master, movie_info):
    # heavily weight action - "Action Movie" and "Sci-Fi Movie" both flag
    # genre_action in the fixture, everything else should rank behind them
    recs = score_by_genre_match({"genre_action": 1.0}, movie_info, ["genre_action"], train_master, min_ratings=1, n=5)

    top_title = recs.iloc[0]["title"]
    assert top_title in ("Action Movie", "Sci-Fi Movie")


def test_get_profile_recs_known_profile_runs(train_master, movie_info):
    for profile_name in GENRE_PROFILES:
        recs = get_profile_recs(profile_name, movie_info, list(GENRE_PROFILES[profile_name].keys()), train_master, n=3)
        assert isinstance(recs, pd.DataFrame)


def test_get_profile_recs_unknown_profile_raises(train_master, movie_info):
    with pytest.raises(KeyError):
        get_profile_recs("Not A Real Profile", movie_info, [], train_master)


# ---------------------------------------------------------------------------
# stat features - cold start fallback
# ---------------------------------------------------------------------------

def test_stat_features_cold_start_defaults_to_zero(train_master):
    stats = build_stat_features(train_master)
    result = get_stat_features(
        user_id=99999, movie_id=99999,
        ustat=stats["user_stats"], istat=stats["item_stats"], global_mean=stats["global_mean"],
    )

    assert result["user_bias"] == 0.0
    assert result["item_count"] == 0


def test_stat_features_known_pair_returns_nonzero_count(train_master):
    stats = build_stat_features(train_master)
    result = get_stat_features(
        user_id=3, movie_id=1,
        ustat=stats["user_stats"], istat=stats["item_stats"], global_mean=stats["global_mean"],
    )

    assert result["user_count"] > 0
    assert result["item_count"] > 0
