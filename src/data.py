"""
Loading and cleaning for the MovieLens-100k data.

Nothing fancy here - read the TSVs, fix up dtypes, merge into the
wide tables the rest of the pipeline works off of.

IMPORTANT: test_master only gets touched during final evaluation.
Don't use it to compute stats, tune anything, or eyeball distributions
before then - that's how leakage sneaks in.
"""

import pandas as pd


def load_raw(data_dir="data"):
    ratings_train = pd.read_csv(f"{data_dir}/ratings_train.tsv", sep="\t")
    ratings_test = pd.read_csv(f"{data_dir}/ratings_test.tsv", sep="\t")
    movie_info = pd.read_csv(f"{data_dir}/movie_info.tsv", sep="\t")
    user_info = pd.read_csv(f"{data_dir}/user_info.tsv", sep="\t")
    return ratings_train, ratings_test, movie_info, user_info


def clean_movies(movie_info):
    movie_info = movie_info.copy()

    # dates come in as '01-Jan-1995', one row has no date at all and gets dropped
    movie_info["release_date"] = pd.to_datetime(
        movie_info["release_date"], format="%d-%b-%Y", errors="coerce"
    )
    movie_info = movie_info.dropna(subset=["release_date"])

    movie_info["release_year"] = movie_info["release_date"].dt.year
    movie_info["decade"] = (movie_info["release_year"] // 10 * 10).astype("Int64").astype(str)

    movie_info = movie_info.rename(columns={"movie_title": "title"})
    return movie_info


def get_genre_cols(movie_info):
    # the 19 binary genre flags all share this prefix
    return [c for c in movie_info.columns if c.startswith("genre_")]


def clean_users(user_info):
    user_info = user_info.copy()
    user_info["age"] = user_info["age"].fillna(user_info["age"].median())
    user_info["occupation"] = user_info["occupation"].fillna("other")
    return user_info


def clean_ratings(ratings_train, ratings_test):
    ratings_train = ratings_train.copy()
    ratings_test = ratings_test.copy()
    ratings_train["date"] = pd.to_datetime(ratings_train["date"], errors="coerce")
    ratings_test["date"] = pd.to_datetime(ratings_test["date"], errors="coerce")
    return ratings_train, ratings_test


def build_master_tables(ratings_train, ratings_test, user_info, movie_info):
    train_master = (
        ratings_train.merge(user_info, on="user_id", how="left")
        .merge(movie_info, on="movie_id", how="left")
    )
    test_master = (
        ratings_test.merge(user_info, on="user_id", how="left")
        .merge(movie_info, on="movie_id", how="left")
    )
    return train_master, test_master


def load_and_clean(data_dir="data"):
    """Runs the whole load -> clean -> merge chain and hands back everything
    downstream code needs, bundled in a dict so callers can grab what they
    want by name instead of unpacking a long tuple."""
    ratings_train, ratings_test, movie_info, user_info = load_raw(data_dir)

    movie_info = clean_movies(movie_info)
    user_info = clean_users(user_info)
    ratings_train, ratings_test = clean_ratings(ratings_train, ratings_test)

    train_master, test_master = build_master_tables(
        ratings_train, ratings_test, user_info, movie_info
    )

    return {
        "ratings_train": ratings_train,
        "ratings_test": ratings_test,
        "movie_info": movie_info,
        "user_info": user_info,
        "train_master": train_master,
        "test_master": test_master,
        "genre_cols": get_genre_cols(movie_info),
    }
