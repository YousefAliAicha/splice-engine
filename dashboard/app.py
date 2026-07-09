"""
Splice dashboard.

Run with:
    streamlit run dashboard/app.py

Layout is a 3-column grid: quiet technical context on the left, the main
interactive panel in the center, a trending-titles strip on the right so
the wide screen doesn't feel empty on either side.

"Advanced" (raw user_id + full ensemble) is intentionally not a top-level
tab - it's a small link tucked into the corner of the panel, since picking
an ID out of 943 isn't how a real visitor would use this.

Every rating - per-movie or end-of-quiz experience rating - gets appended
to feedback/ratings_log.csv for the analysis notebook to pick up later.
"""

import csv
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import load_and_clean
from src.engines import (
    recommend,
    build_ensemble_predict,
    get_profile_recs,
    score_by_genre_match,
    GENRE_PROFILES,
)

FEEDBACK_DIR = Path(__file__).resolve().parent.parent / "feedback"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return load_and_clean()


@st.cache_resource
def get_models():
    with open(MODELS_DIR / "base_models.pkl", "rb") as f:
        base_models = pickle.load(f)
    with open(MODELS_DIR / "meta_learner.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(MODELS_DIR / "popularity_lists.pkl", "rb") as f:
        popularity_lists = pickle.load(f)
    with open(MODELS_DIR / "knn.pkl", "rb") as f:
        knn = pickle.load(f)
    with open(MODELS_DIR / "stat_features.pkl", "rb") as f:
        stat_features = pickle.load(f)

    ensemble_predict = build_ensemble_predict(
        base_models, meta, stat_features["user_stats"], stat_features["item_stats"], stat_features["global_mean"]
    )
    return popularity_lists, knn, ensemble_predict


def load_results():
    """Pulls the Test RMSE out of results.txt. Returns None if it's not
    there yet (e.g. training hasn't been run)."""
    results_path = MODELS_DIR / "results.txt"
    if not results_path.exists():
        return None
    text = results_path.read_text()
    for line in text.splitlines():
        if line.startswith("Test RMSE:"):
            return float(line.split(":")[1].strip())
    return None


def load_eval_history():
    path = MODELS_DIR / "eval_history.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
    FEEDBACK_DIR.mkdir(exist_ok=True)
    path = FEEDBACK_DIR / "ratings_log.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(fields)


# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------

CINEMA_CSS = """
<style>
    .stApp {
        background: radial-gradient(ellipse at top, #1c1512 0%, #0f0c0a 70%);
    }

    /* header */
    .splice-header { text-align: center; padding: 2rem 0 1.2rem 0; }
    .splice-header h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.5rem;
        letter-spacing: 0.14em;
        color: #d9b869;
        margin-bottom: 0.15rem;
        text-transform: uppercase;
    }
    .splice-header p {
        color: #8a7c6c;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    }

    /* center the tab row */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 2.5rem;
        border-bottom: 1px solid #3a2f26;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8a7c6c;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        transition: color 0.3s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        color: #d9b869 !important;
    }

    /* side columns - quiet, low-opacity supporting content */
    .side-panel { opacity: 0.55; padding-top: 1.2rem; }
    .side-label {
        color: #d9b869;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .side-fact {
        color: #8a7c6c;
        font-size: 0.78rem;
        line-height: 1.7;
        margin-bottom: 0.9rem;
        border-left: 2px solid #3a2f26;
        padding-left: 0.7rem;
    }
    .trending-item {
        color: #9c8f7d;
        font-size: 0.8rem;
        padding: 0.4rem 0;
        border-bottom: 1px solid #241d18;
    }

    /* recommendation rows */
    .movie-row {
        background: #181310;
        border-left: 3px solid #7a5c2e;
        border-radius: 4px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        transition: all 0.3s ease-in-out;
    }
    .movie-row:hover { border-left-color: #d9b869; }
    .movie-row .rank { color: #6b5d4d; font-size: 0.8rem; letter-spacing: 0.08em; }
    .movie-row .title { color: #ece4d8; font-size: 1.05rem; font-family: Georgia, serif; }

    .section-label {
        color: #d9b869;
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        border-bottom: 1px solid #3a2f26;
        padding-bottom: 0.4rem;
        margin: 1.6rem 0 1rem 0;
    }

    /* genre / profile buttons get a hover lift + glow */
    div[data-testid="stButton"] > button {
        transition: all 0.3s ease-in-out;
        border: 1px solid #3a2f26;
    }
    div[data-testid="stButton"] > button:hover {
        transform: scale(1.03);
        border-color: #d9b869;
        box-shadow: 0 0 12px rgba(217, 184, 105, 0.25);
    }

    /* quiz progress bar */
    .quiz-progress-track {
        background: #241d18;
        border-radius: 4px;
        height: 4px;
        margin-bottom: 1.6rem;
        overflow: hidden;
    }
    .quiz-progress-fill {
        background: #d9b869;
        height: 100%;
        transition: width 0.4s ease-in-out;
    }

    /* quiz question fade-in */
    .quiz-question {
        animation: fadeInSlide 0.35s ease-in-out;
    }
    @keyframes fadeInSlide {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .advanced-link {
        text-align: right;
        color: #5c5346;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        margin-top: 2rem;
    }
</style>
"""

st.set_page_config(page_title="Splice", page_icon=None, layout="wide")
st.markdown(CINEMA_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="splice-header">
        <h1>Splice</h1>
        <p>Movie Recommender Engine &nbsp;&middot;&nbsp; by Yousef Ali Aicha</p>
    </div>
    """,
    unsafe_allow_html=True,
)

data = get_data()
train_master = data["train_master"]
movie_info = data["movie_info"]
user_info = data["user_info"]
genre_cols = data["genre_cols"]
popularity_lists, knn, ensemble_predict = get_models()


def render_recs(recs_df, engine_label, context_key):
    st.markdown('<div class="section-label">Recommended for you</div>', unsafe_allow_html=True)
    for i, row in recs_df.reset_index(drop=True).iterrows():
        st.markdown(
            f"""
            <div class="movie-row">
                <div class="rank">{str(i + 1).zfill(2)}</div>
                <div class="title">{row['title']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns([4, 1])
        with cols[0]:
            rating = st.slider("Rate", 0, 10, 5, key=f"{context_key}_slider_{i}", label_visibility="collapsed")
        with cols[1]:
            if st.button("Log", key=f"{context_key}_log_{i}"):
                log_row(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    movie_id=int(row["movie_id"]), title=row["title"],
                    engine_used=engine_label, user_rating=rating,
                )
                st.toast(f"Logged {rating}/10 for {row['title']}")


# ---------------------------------------------------------------------------
# left / center / right grid
# ---------------------------------------------------------------------------

left_col, center_col, right_col = st.columns([1.2, 2.5, 1.3])

with left_col:
    if "rmse_val" not in st.session_state:
        st.session_state["rmse_val"] = load_results()

    st.markdown(
        """
        <div class="side-panel">
            <div class="side-label">Under the hood</div>
            <div class="side-fact">Dataset: MovieLens 100k<br>943 users &middot; 1,681 films</div>
            <div class="side-fact">Collaborative filtering ensemble<br>SVD, item-KNN, Baseline,<br>SlopeOne, Co-Clustering</div>
            <div class="side-fact">Meta-learner: LightGBM<br>trained on out-of-fold predictions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state["rmse_val"] is not None:
        st.metric("Test RMSE", f"{st.session_state['rmse_val']:.4f}")
    else:
        st.caption("No results.txt found yet - run `python -m src.train` first.")

    eval_history = load_eval_history()
    if eval_history and "train" in eval_history and "valid" in eval_history:
        curve_df = pd.DataFrame({
            "train": eval_history["train"]["rmse"],
            "valid": eval_history["valid"]["rmse"],
        })
        st.caption("Meta-learner boosting curve")
        st.line_chart(curve_df, height=140)

with right_col:
    trending = popularity_lists["overall"].head(8)["title"].tolist()
    items_html = "".join(f'<div class="trending-item">{t}</div>' for t in trending)
    st.markdown(
        f"""
        <div class="side-panel">
            <div class="side-label">Trending on Splice</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

with center_col:
    tab_profiles, tab_quiz = st.tabs(["Browse by taste", "Take the quiz"])

    with tab_profiles:
        st.markdown('<div class="section-label">Pick what you are in the mood for</div>', unsafe_allow_html=True)

        profile_cols = st.columns(2)
        for i, name in enumerate(GENRE_PROFILES):
            with profile_cols[i % 2]:
                if st.button(name, key=f"profile_{name}", use_container_width=True):
                    st.session_state["chosen_profile"] = name

        active_profile = st.session_state.get("chosen_profile")
        if active_profile:
            st.write("")
            recs = get_profile_recs(active_profile, movie_info, genre_cols, train_master, n=10)
            render_recs(recs, engine_label=f"profile:{active_profile}", context_key="profile")
            st.caption(
                "Cold start: genre-weighted, catalog-wide scoring - no rating "
                "history required. This is the same fallback the ensemble uses "
                "internally for brand-new users."
            )

    with tab_quiz:
        QUIZ_STEPS = ["mood", "era", "genres"]
        if "quiz_step" not in st.session_state:
            st.session_state["quiz_step"] = 0
        if "quiz_answers" not in st.session_state:
            st.session_state["quiz_answers"] = {}

        step = st.session_state["quiz_step"]
        progress_pct = int((step / len(QUIZ_STEPS)) * 100)

        st.markdown(
            f"""
            <div class="quiz-progress-track">
                <div class="quiz-progress-fill" style="width: {progress_pct}%;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if step < len(QUIZ_STEPS):
            st.markdown('<div class="quiz-question">', unsafe_allow_html=True)

            if QUIZ_STEPS[step] == "mood":
                st.markdown('<div class="section-label">What kind of night is this?</div>', unsafe_allow_html=True)
                answer = st.radio(
                    "mood", ["Something exciting", "Something heartfelt", "Something funny", "Something dark"],
                    label_visibility="collapsed", key="mood_radio",
                )
                if st.button("Next", key="next_mood"):
                    st.session_state["quiz_answers"]["mood"] = answer
                    st.session_state["quiz_step"] += 1
                    st.rerun()

            elif QUIZ_STEPS[step] == "era":
                st.markdown('<div class="section-label">Old or new?</div>', unsafe_allow_html=True)
                answer = st.radio(
                    "era", ["Doesn't matter", "Give me a classic", "Keep it modern"],
                    label_visibility="collapsed", key="era_radio",
                )
                nav_cols = st.columns(2)
                with nav_cols[0]:
                    if st.button("Back", key="back_era"):
                        st.session_state["quiz_step"] -= 1
                        st.rerun()
                with nav_cols[1]:
                    if st.button("Next", key="next_era"):
                        st.session_state["quiz_answers"]["era"] = answer
                        st.session_state["quiz_step"] += 1
                        st.rerun()

            elif QUIZ_STEPS[step] == "genres":
                st.markdown('<div class="section-label">Any genres you\'re drawn to?</div>', unsafe_allow_html=True)
                answer = st.multiselect(
                    "genres",
                    ["Action", "Adventure", "Comedy", "Crime", "Drama", "Fantasy",
                     "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller"],
                    label_visibility="collapsed", key="genres_multi",
                )
                nav_cols = st.columns(2)
                with nav_cols[0]:
                    if st.button("Back", key="back_genres"):
                        st.session_state["quiz_step"] -= 1
                        st.rerun()
                with nav_cols[1]:
                    if st.button("Show me something", key="finish_quiz"):
                        st.session_state["quiz_answers"]["genres"] = answer
                        st.session_state["quiz_step"] += 1
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            answers = st.session_state["quiz_answers"]
            weights = {}

            mood_map = {
                "Something exciting": {"genre_action": 1.0, "genre_adventure": 0.8, "genre_thriller": 0.6},
                "Something heartfelt": {"genre_drama": 1.0, "genre_romance": 0.7},
                "Something funny": {"genre_comedy": 1.0},
                "Something dark": {"genre_horror": 0.8, "genre_thriller": 0.8, "genre_crime": 0.6},
            }
            for genre, w in mood_map[answers["mood"]].items():
                weights[genre] = weights.get(genre, 0) + w
            for g in answers.get("genres", []):
                key = f"genre_{g.lower().replace('-', '_')}"
                weights[key] = weights.get(key, 0) + 1.0

            recs = score_by_genre_match(weights, movie_info, genre_cols, train_master, n=10)

            if answers["era"] == "Give me a classic":
                older = movie_info[movie_info["release_year"] < 1980][["movie_id"]]
                merged = recs.merge(older, on="movie_id", how="inner")
                recs = merged if not merged.empty else recs
            elif answers["era"] == "Keep it modern":
                newer = movie_info[movie_info["release_year"] >= 1990][["movie_id"]]
                merged = recs.merge(newer, on="movie_id", how="inner")
                recs = merged if not merged.empty else recs

            render_recs(recs, engine_label="quiz", context_key="quiz")

            st.caption(
                "Cold start: this is a new session with no rating history, so these "
                "come from genre-affinity scoring against the catalog, not the "
                "trained collaborative-filtering ensemble. See Advanced Mode to try "
                "the full ensemble against an existing user's rating history."
            )

            st.markdown('<div class="section-label">How did we do?</div>', unsafe_allow_html=True)
            experience = st.slider("Overall, how well did this match what you wanted?", 0, 10, 5, key="quiz_experience")
            if st.button("Submit experience rating", key="submit_experience"):
                log_row(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    movie_id="", title="", engine_used="quiz_experience", user_rating=experience,
                )
                st.toast("Thanks, logged your experience rating")

            if st.button("Retake the quiz", key="retake_quiz"):
                st.session_state["quiz_step"] = 0
                st.session_state["quiz_answers"] = {}
                st.rerun()

    # discreet advanced-mode link, tucked at the bottom of the panel
    st.markdown('<div class="advanced-link">Advanced Mode</div>', unsafe_allow_html=True)
    with st.expander(" ", expanded=False):
        st.caption(
            "The dataset has 943 users with real rating histories. This runs the "
            "actual routing logic (popularity / KNN / full ensemble) against one "
            "of them directly, mainly to demonstrate the engine-selection behavior."
        )

        st.caption(
            "Full retraining (10-fold CV across five models) takes several minutes "
            "and isn't triggered from the UI - run `python -m src.train` and click "
            "below to pull the fresh numbers in without restarting the app."
        )
        if st.button("Refresh metrics from disk", key="refresh_metrics"):
            st.session_state["rmse_val"] = load_results()
            st.rerun()
        user_ids = sorted(user_info["user_id"].unique())
        user_id = st.selectbox("User ID", user_ids, key="advanced_user_id")

        rating_count = int((train_master["user_id"] == user_id).sum())
        engine_name = "popularity" if rating_count == 0 else "knn" if rating_count < 5 else "ensemble"
        st.write(f"{rating_count} ratings on file, routed to the **{engine_name}** engine.")

        if st.button("Run recommendation", key="run_advanced"):
            recs = recommend(
                user_id, train_master, movie_info, popularity_lists, knn,
                ensemble_predict, user_info=user_info, n=10,
            )
            recs_df = pd.DataFrame(recs, columns=["movie_id", "title", "score"])
            st.session_state["advanced_recs"] = recs_df
            st.session_state["advanced_engine"] = engine_name

        if "advanced_recs" in st.session_state:
            render_recs(
                st.session_state["advanced_recs"],
                engine_label=st.session_state["advanced_engine"],
                context_key="advanced",
            )
