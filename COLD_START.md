## Handling cold start

A collaborative-filtering ensemble is only as good as the rating history it
has to work with. A brand-new user - which is every visitor to the quiz or
the taste-profile browser - has none, so scoring them through SVD, item-KNN,
or the LightGBM meta-learner isn't just weaker, it's meaningless: those
models have no signal for a user_id they've never seen.

Splice routes on rating count rather than forcing every user through one
model:

| Ratings on file | Engine used | Why |
|---|---|---|
| 0 | Genre-affinity scoring | No collaborative signal exists yet. Falls back to content (genre) matching weighted by how well-liked a movie is overall. |
| 1-4 | Item-based KNN | Too little history for the ensemble's stat features (user bias, std, count) to be reliable, but enough for "movies similar to the one you just rated highly" to work. |
| 5+ | Full stacked ensemble | Enough history for user-level bias and the meta-learner's blending to actually mean something. |

The quiz and taste-profile browser are both cold-start by definition - a
first-time visitor has no account and no history - so they run entirely on
the genre-affinity path (`src/engines.py::score_by_genre_match`). This is the
same fallback the router uses internally for any real user with zero ratings
(`src/engines.py::recommend`), not a separate simplified demo path. The
dashboard states this explicitly under quiz/profile results rather than
presenting genre-matched picks as if they came from the trained ensemble.

The "Advanced" panel in the dashboard exists specifically to demonstrate the
full ensemble against a user who *does* have rating history, since that's
not something a first-time visitor can trigger on their own.
