"""
Builds content embeddings from TMDB synopses and a FAISS index over them.

This is the content-based half of the hybrid system: everything else in
src/engines.py works off collaborative signal (who rated what) or genre
flags. This module adds an actual understanding of what a movie is *about*,
via sentence embeddings of its synopsis, so "similar plot" becomes a real
computable thing rather than "shares a genre tag."

Run once per dataset to build the index:
    python -m src.embeddings --dataset ml-100k

Needs a TMDB key (src/config.py) since synopses come from there, not
MovieLens. Movies with no TMDB match get skipped - not every obscure title
in the catalog has a match, and a hybrid system should degrade gracefully
for those rather than fail.
"""

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.tmdb import fetch_many

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs fine on CPU - no reason
                                   # to reach for something bigger at this catalog size


def build_content_index(movie_info, output_dir):
    """Embeds every movie's TMDB synopsis and writes a FAISS index plus the
    movie_id ordering that maps index rows back to actual movies."""
    print(f"Fetching synopses for {len(movie_info)} movies from TMDB...")
    tmdb_info = fetch_many(movie_info["title"].tolist())

    movie_ids, texts = [], []
    for _, row in movie_info.iterrows():
        info = tmdb_info.get(row["title"])
        if info and info.get("overview"):
            movie_ids.append(row["movie_id"])
            texts.append(info["overview"])

    skipped = len(movie_info) - len(movie_ids)
    print(f"{len(movie_ids)} movies have a synopsis, {skipped} skipped (no TMDB match or empty overview).")

    print(f"Embedding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # normalized embeddings + inner product = cosine similarity, cheaper
    # than L2 distance and exactly what we want for "how similar in meaning"
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "content_index.faiss"))
    with open(output_dir / "content_movie_ids.pkl", "wb") as f:
        pickle.dump(movie_ids, f)
    with open(output_dir / "content_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved content index to {output_dir}/")
    return index, movie_ids, embeddings


def load_content_index(models_dir):
    """Returns (index, movie_ids) or (None, None) if the index hasn't been
    built yet - callers should treat a missing index as "no content signal
    available" rather than crash, since hybridization is additive, not a
    hard requirement for the rest of the system to work."""
    index_path = models_dir / "content_index.faiss"
    ids_path = models_dir / "content_movie_ids.pkl"

    if not index_path.exists() or not ids_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    with open(ids_path, "rb") as f:
        movie_ids = pickle.load(f)
    return index, movie_ids


def get_similar_by_plot(movie_id, index, content_movie_ids, movie_info, n=10, min_similarity=0.47):
    """Movies with the most similar synopsis embedding - genuinely
    content-based, distinct from the collaborative KNN engine in
    engines.py (which finds movies rated similarly by the same users,
    regardless of what either movie is actually about).

    min_similarity filters out weak matches. On a catalog this size with
    short TMDB synopses, once genuinely similar movies run out, FAISS still
    returns *something* as the "nearest" neighbor - just not a meaningfully
    similar one. Threshold was set from a real measurement (see ISSUES.md):
    querying GoodFellas, genuine matches (Godfather I/II) scored 0.48-0.49
    while vocabulary/tone-only coincidences scored ~0.45. 0.47 sits between
    those two clusters. This is a judgment call from one investigated
    example, not a formally tuned value - a different query might reveal
    the boundary sits elsewhere.
    """
    if index is None or movie_id not in content_movie_ids:
        return []

    query_idx = content_movie_ids.index(movie_id)
    query_vector = index.reconstruct(query_idx).reshape(1, -1)

    k = min(n + 1, index.ntotal)
    scores, neighbor_idxs = index.search(query_vector, k)
    scores, neighbor_idxs = scores[0], neighbor_idxs[0]

    titles_lookup = movie_info.set_index("movie_id")["title"]
    results = []
    for score, idx in zip(scores, neighbor_idxs):
        if idx == query_idx or score < min_similarity:
            continue
        results.append(titles_lookup.get(content_movie_ids[idx], "Unknown"))
        if len(results) == n:
            break
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-100k")
    args = parser.parse_args()

    if args.dataset == "ml-1m":
        from src.data_1m import load_and_clean
    else:
        from src.data import load_and_clean

    data = load_and_clean()
    output_dir = Path("models") / args.dataset
    build_content_index(data["movie_info"], output_dir)
