"""
Diagnostic for the content-embedding similarity feature (src/embeddings.py).

Prints a movie's actual TMDB synopsis alongside its nearest neighbors' -
side by side, so a confusing "why did X get matched to Y" result can be
inspected directly rather than guessed at. Also prints raw cosine scores,
since "5th nearest neighbor" can still mean "not very similar at all" if
the actual similarity score is low.

Run with:
    python -m analysis.inspect_embeddings --dataset ml-100k --title "GoodFellas"
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from src.embeddings import load_content_index
from src.tmdb import fetch_many


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-100k")
    parser.add_argument("--title", required=True, help="Movie title substring to look up, e.g. 'GoodFellas'")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    if args.dataset == "ml-1m":
        from src.data_1m import load_and_clean
    else:
        from src.data import load_and_clean

    data = load_and_clean()
    movie_info = data["movie_info"]

    models_dir = Path("models") / args.dataset
    index, content_movie_ids = load_content_index(models_dir)
    if index is None:
        print(f"No content index found at {models_dir}. Run `python -m src.embeddings --dataset {args.dataset}` first.")
        return

    with open(models_dir / "content_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    matches = movie_info[movie_info["title"].str.contains(args.title, case=False, na=False)]
    if matches.empty:
        print(f"No movie matching '{args.title}' found.")
        return

    movie_id = matches.iloc[0]["movie_id"]
    query_title = matches.iloc[0]["title"]

    if movie_id not in content_movie_ids:
        print(f"'{query_title}' has no content embedding (no TMDB synopsis match).")
        return

    query_idx = content_movie_ids.index(movie_id)
    query_vector = index.reconstruct(query_idx).reshape(1, -1)

    k = min(args.n + 1, index.ntotal)
    scores, neighbor_idxs = index.search(query_vector, k)
    scores, neighbor_idxs = scores[0], neighbor_idxs[0]

    neighbor_titles = [movie_info.loc[movie_info["movie_id"] == content_movie_ids[i], "title"].values[0] for i in neighbor_idxs]
    tmdb_info = fetch_many([query_title] + neighbor_titles)

    print(f"\n=== Query: {query_title} ===")
    print(tmdb_info.get(query_title, {}).get("overview", "(no synopsis)"))
    print()

    for score, idx, title in zip(scores, neighbor_idxs, neighbor_titles):
        if idx == query_idx:
            continue
        print(f"--- {title}  (cosine similarity: {score:.3f}) ---")
        print(tmdb_info.get(title, {}).get("overview", "(no synopsis)"))
        print()


if __name__ == "__main__":
    main()
