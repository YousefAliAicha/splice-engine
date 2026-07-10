"""
Thin wrapper around the TMDB API for pulling posters and synopses to make
the dashboard look like a real product instead of a bare list of titles.

Everything gets cached to disk (models/tmdb_cache.json) since MovieLens
titles don't change - no reason to re-hit the API for the same movie every
time someone loads the dashboard. TMDB's free tier rate limit is generous
but there's still no reason to burn requests we don't need to.
"""

import json
import re
from pathlib import Path

import requests

from src.config import get_tmdb_key

TMDB_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

CACHE_PATH = Path(__file__).resolve().parent.parent / "models" / "tmdb_cache.json"


def _load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=1)


def _extract_year(title):
    """MovieLens titles look like 'Toy Story (1995)' - pull the year out so
    the TMDB search can disambiguate remakes/sequels with the same name."""
    match = re.search(r"\((\d{4})\)", title)
    return match.group(1) if match else None


def _clean_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def fetch_movie_info(movielens_title):
    """Returns {'poster_url': ..., 'overview': ...} for a MovieLens title,
    or None if TMDB has no match. Cached after the first lookup."""
    cache = _load_cache()

    if movielens_title in cache:
        return cache[movielens_title]

    query_title = _clean_title(movielens_title)
    year = _extract_year(movielens_title)

    params = {"api_key": get_tmdb_key(), "query": query_title}
    if year:
        params["year"] = year

    response = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=10)
    response.raise_for_status()
    results = response.json().get("results", [])

    if not results:
        cache[movielens_title] = None
        _save_cache(cache)
        return None

    top = results[0]
    info = {
        "poster_url": f"{IMAGE_BASE}{top['poster_path']}" if top.get("poster_path") else None,
        "overview": top.get("overview", ""),
        "tmdb_id": top.get("id"),
    }

    cache[movielens_title] = info
    _save_cache(cache)
    return info


def fetch_many(titles):
    """Batch version - same caching, just saves once at the end instead of
    per-title so a first-run warm-up of the whole catalog doesn't write to
    disk hundreds of times."""
    cache = _load_cache()
    results = {}

    for title in titles:
        if title in cache:
            results[title] = cache[title]
            continue

        query_title = _clean_title(title)
        year = _extract_year(title)
        params = {"api_key": get_tmdb_key(), "query": query_title}
        if year:
            params["year"] = year

        try:
            response = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=10)
            response.raise_for_status()
            hits = response.json().get("results", [])
        except requests.RequestException:
            hits = []

        if hits:
            top = hits[0]
            info = {
                "poster_url": f"{IMAGE_BASE}{top['poster_path']}" if top.get("poster_path") else None,
                "overview": top.get("overview", ""),
                "tmdb_id": top.get("id"),
            }
        else:
            info = None

        cache[title] = info
        results[title] = info

    _save_cache(cache)
    return results
