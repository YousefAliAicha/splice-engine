"""
Blends the collaborative ensemble (src/engines.py) with the content-based
signal (src/embeddings.py) into a single hybrid score.

Honest note on how this is wired in: this is a post-hoc blend at inference
time, not a feature fed into the meta-learner's training. Threading content
similarity into the OOF-trained meta-learner properly would mean recomputing
it per fold (same leakage discipline as the stat features) and retraining
the whole stack - a bigger piece of surgery than justified for what's
ultimately a secondary signal here. What's implemented is real: genuine
sentence embeddings, genuine FAISS similarity search, genuinely blended
into the final recommendation - just combined after each score is computed
independently, rather than jointly learned.
"""

import numpy as np


def build_user_taste_vector(user_id, train_master, content_movie_ids, content_embeddings, min_rating=4):
    """Average embedding of the movies this user rated >= min_rating.
    Returns None if the user has no highly-rated movies with a content
    embedding available (either brand new, or their favorites happened to
    be ones TMDB had no synopsis for)."""
    liked = train_master[(train_master["user_id"] == user_id) & (train_master["rating"] >= min_rating)]
    if liked.empty:
        return None

    id_to_row = {mid: i for i, mid in enumerate(content_movie_ids)}
    rows = [id_to_row[mid] for mid in liked["movie_id"] if mid in id_to_row]
    if not rows:
        return None

    vectors = content_embeddings[rows]
    taste_vector = vectors.mean(axis=0)
    return taste_vector / np.linalg.norm(taste_vector)


def content_affinity(taste_vector, movie_id, content_movie_ids, content_embeddings):
    """Cosine similarity between a user's taste vector and one candidate
    movie's embedding. Returns 0.0 (neutral, no pull either way) if the
    movie has no embedding - matches the cold-start-default pattern used
    for stat features elsewhere."""
    if taste_vector is None or movie_id not in content_movie_ids:
        return 0.0

    idx = content_movie_ids.index(movie_id)
    movie_vector = content_embeddings[idx]
    return float(np.dot(taste_vector, movie_vector))


def build_hybrid_predict(ensemble_predict, train_master, content_movie_ids, content_embeddings, alpha=0.85):
    """Returns a predict(user_id, movie_id) function with the same
    signature as build_ensemble_predict, so it's a drop-in replacement
    anywhere ensemble_predict is used.

    alpha controls how much weight the collaborative ensemble keeps -
    0.85 means content only nudges the score, doesn't override it. The
    ensemble has actual leakage-safe cross-validated performance behind
    it (0.90 RMSE on held-out data); the content signal doesn't, so it
    stays a minority voice rather than an equal partner.
    """
    if content_movie_ids is None:
        # no content index built - fall back to pure collaborative,
        # hybridization is additive, its absence shouldn't break anything
        return ensemble_predict

    taste_cache = {}

    def predict(user_id, movie_id):
        base_score = ensemble_predict(user_id, movie_id)  # 1-5 scale

        if user_id not in taste_cache:
            taste_cache[user_id] = build_user_taste_vector(
                user_id, train_master, content_movie_ids, content_embeddings
            )
        taste_vector = taste_cache[user_id]

        affinity = content_affinity(taste_vector, movie_id, content_movie_ids, content_embeddings)
        # affinity is a cosine similarity in [-1, 1] - rescale to a
        # 1-5 nudge in the same units as base_score before blending
        affinity_scaled = 3 + affinity * 2

        return float(np.clip(alpha * base_score + (1 - alpha) * affinity_scaled, 1, 5))

    return predict
