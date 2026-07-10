"""
Loader for MovieLens-1M (the .dat files - users.dat, movies.dat, ratings.dat).

1M's format is different enough from the 100k .tsv files that this isn't a
small tweak to data.py - different delimiter (:: instead of tab), different
columns, ages come pre-binned into age-group codes instead of raw numbers,
occupation is a numeric code instead of a string, and there's no pre-made
train/test split.

The output of load_and_clean() here matches the same dict shape data.py
produces, so nothing downstream (features.py, engines.py) needs to know or
care which dataset it's looking at.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import get_genre_cols  # reused as-is, just checks for genre_ prefix


# 1M encodes occupation as an integer 0-20 rather than a string - this
# mapping comes straight from the dataset's own README
OCCUPATION_MAP = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}

# genre names as they appear in movies.dat, mapped to the genre_ prefix
# convention used throughout the rest of the codebase
GENRE_NAME_MAP = {
    "Action": "genre_action", "Adventure": "genre_adventure", "Animation": "genre_animation",
    "Children's": "genre_childrens", "Comedy": "genre_comedy", "Crime": "genre_crime",
    "Documentary": "genre_documentary", "Drama": "genre_drama", "Fantasy": "genre_fantasy",
    "Film-Noir": "genre_film_noir", "Horror": "genre_horror", "Musical": "genre_musical",
    "Mystery": "genre_mystery", "Romance": "genre_romance", "Sci-Fi": "genre_sci_fi",
    "Thriller": "genre_thriller", "War": "genre_war", "Western": "genre_western",
}


def load_raw(data_dir="data/ml-1m"):
    # latin-1 because the original dataset predates consistent UTF-8 usage
    # and a handful of titles have accented characters that break otherwise
    users = pd.read_csv(
        f"{data_dir}/users.dat", sep="::", engine="python", encoding="latin-1",
        names=["user_id", "gender", "age_code", "occupation_code", "zip"],
    )
    movies = pd.read_csv(
        f"{data_dir}/movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["movie_id", "title_raw", "genres_raw"],
    )
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return users, movies, ratings


def clean_users(users):
    users = users.copy()
    users["occupation"] = users["occupation_code"].map(OCCUPATION_MAP)
    # age_code is already binned (1, 18, 25, 35, 45, 50, 56 = age-group
    # markers, not real ages) - keep it as-is under the same "age" column
    # name the 100k pipeline uses, just document what it actually means
    users["age"] = users["age_code"]
    return users[["user_id", "gender", "age", "occupation"]]


def clean_movies(movies):
    movies = movies.copy()

    # "Toy Story (1995)" -> title="Toy Story", release_year=1995
    year_extracted = movies["title_raw"].str.extract(r"\((\d{4})\)\s*$")
    movies["release_year"] = pd.to_numeric(year_extracted[0], errors="coerce")
    movies["title"] = movies["title_raw"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["decade"] = (movies["release_year"] // 10 * 10).astype("Int64").astype(str)
    movies = movies.dropna(subset=["release_year"])

    genre_lists = movies["genres_raw"].str.split("|")
    for raw_name, col_name in GENRE_NAME_MAP.items():
        movies[col_name] = genre_lists.apply(lambda g: int(raw_name in g))

    keep_cols = ["movie_id", "title", "release_year", "decade"] + list(GENRE_NAME_MAP.values())
    return movies[keep_cols]


def split_ratings(ratings, test_size=0.2, random_state=42):
    """1M doesn't ship a pre-made train/test split like 100k does - just a
    random split, stratified by nothing in particular. Good enough for
    benchmarking; a time-based split would be more realistic but adds
    complexity this doesn't need."""
    return train_test_split(ratings, test_size=test_size, random_state=random_state)


def build_master_tables(ratings_train, ratings_test, users, movies):
    train_master = ratings_train.merge(users, on="user_id", how="left").merge(movies, on="movie_id", how="left")
    test_master = ratings_test.merge(users, on="user_id", how="left").merge(movies, on="movie_id", how="left")
    return train_master, test_master


def load_and_clean(data_dir="data/ml-1m"):
    users, movies, ratings = load_raw(data_dir)

    users = clean_users(users)
    movies = clean_movies(movies)
    ratings_train, ratings_test = split_ratings(ratings)

    train_master, test_master = build_master_tables(ratings_train, ratings_test, users, movies)

    return {
        "ratings_train": ratings_train,
        "ratings_test": ratings_test,
        "movie_info": movies,
        "user_info": users,
        "train_master": train_master,
        "test_master": test_master,
        "genre_cols": get_genre_cols(movies),
    }
