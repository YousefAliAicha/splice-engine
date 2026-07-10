"""
Analyzes feedback/ratings_log.csv - the ratings people actually gave through
the dashboard - and asks a few honest questions:

  - When the ensemble predicts a score, how close is it to what people
    actually rated? (calibration)
  - Do some engines get better received than others? (popularity vs quiz
    vs the trained ensemble)
  - Are quiz results actually landing well? (the end-of-quiz experience score)
  - Any trend over time as more feedback comes in?

Run with:
    python -m analysis.feedback_analysis

Needs at least a handful of logged ratings to say anything meaningful - with
under ~10 rows most of these plots will look like noise, which is expected,
not a bug.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_FEEDBACK_PATH = Path(__file__).resolve().parent.parent / "feedback" / "ratings_log.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def load_feedback(path):
    if not path.exists():
        raise FileNotFoundError(
            f"No feedback logged yet at {path}. "
            "Run the dashboard and rate a few recommendations first, "
            "or run scripts/generate_demo_data.py for synthetic data."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="coerce")
    return df


def plot_calibration(df, save_path):
    """Predicted (1-5 scale) vs actual (0-10 scale) - only rows where the
    ensemble actually produced a score, i.e. the advanced/ensemble path.
    Profile and quiz recs use genre scoring, not a rating prediction, so
    they don't belong in a calibration check."""
    calibrated = df.dropna(subset=["predicted_score"])
    if calibrated.empty:
        print("No ensemble-predicted rows yet - try the Advanced tab in the dashboard.")
        return

    # put both on the same 0-10 scale so the diagonal actually means something
    predicted_scaled = calibrated["predicted_score"] * 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predicted_scaled, calibrated["user_rating"], alpha=0.6, color="#7a5c2e")
    ax.plot([0, 10], [0, 10], linestyle="--", color="#999", label="perfect calibration")
    ax.set_xlabel("Predicted rating (scaled to /10)")
    ax.set_ylabel("Actual user rating (/10)")
    ax.set_title("Ensemble calibration: predicted vs actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_engine_comparison(df, save_path):
    """Average rating by engine - are people happier with the trained
    ensemble, or does simple genre matching hold up just as well? This is
    the actual point of collecting feedback in the first place."""
    per_movie = df[df["engine_used"] != "quiz_experience"].copy()
    if per_movie.empty:
        print("No per-movie ratings yet.")
        return

    # collapse "profile:Romance", "profile:Comedy" etc down to one "profile" bucket
    per_movie["engine_group"] = per_movie["engine_used"].apply(
        lambda x: "profile" if str(x).startswith("profile:") else x
    )

    grouped = per_movie.groupby("engine_group")["user_rating"].agg(["mean", "count"])
    grouped = grouped.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(grouped.index, grouped["mean"], color="#7a5c2e")
    for bar, count in zip(bars, grouped["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"n={count}", ha="center", fontsize=8)
    ax.set_ylabel("Average user rating (/10)")
    ax.set_ylim(0, 10)
    ax.set_title("Average rating by recommendation engine")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_quiz_experience_trend(df, save_path):
    """The end-of-quiz 'how well did this match what you wanted' rating,
    over time - are quiz results getting better received as we iterate,
    or is this flat?"""
    experience = df[df["engine_used"] == "quiz_experience"].sort_values("timestamp")
    if experience.empty:
        print("No quiz experience ratings yet.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(experience["timestamp"], experience["user_rating"], marker="o", color="#7a5c2e")
    ax.set_ylabel("Experience rating (/10)")
    ax.set_xlabel("Time")
    ax.set_ylim(0, 10)
    ax.set_title("Quiz experience rating over time")

    if len(experience) < 3:
        # matplotlib auto-scales the x-axis to a huge multi-year range when
        # there's only a point or two, which looks like a broken/empty
        # chart rather than "not much data yet" - pad by a few minutes
        # around the actual timestamps instead
        pad = pd.Timedelta(minutes=10)
        ax.set_xlim(experience["timestamp"].min() - pad, experience["timestamp"].max() + pad)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_rating_distribution(df, save_path):
    """Simple sanity-check histogram - are people mostly rating things
    high, low, or all over the place? Useful for spotting a rating scale
    that's being used oddly (e.g. everyone just clicking 5 and moving on)."""
    per_movie = df[df["engine_used"] != "quiz_experience"]
    if per_movie.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_movie["user_rating"], bins=range(0, 12), color="#7a5c2e", edgecolor="#181310", align="left")
    ax.set_xlabel("Rating given (/10)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of ratings given")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Path to a feedback CSV (default: feedback/ratings_log.csv)")
    parser.add_argument("--demo", action="store_true", help="Shortcut for --file feedback/demo_ratings_log.csv")
    args = parser.parse_args()

    if args.demo:
        feedback_path = Path(__file__).resolve().parent.parent / "feedback" / "demo_ratings_log.csv"
        figures_dir = FIGURES_DIR / "demo"
    elif args.file:
        feedback_path = Path(args.file)
        figures_dir = FIGURES_DIR
    else:
        feedback_path = DEFAULT_FEEDBACK_PATH
        figures_dir = FIGURES_DIR

    df = load_feedback(feedback_path)
    print(f"Loaded {len(df)} feedback rows from {feedback_path}.")

    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_calibration(df, figures_dir / "calibration.png")
    plot_engine_comparison(df, figures_dir / "engine_comparison.png")
    plot_quiz_experience_trend(df, figures_dir / "quiz_experience_trend.png")
    plot_rating_distribution(df, figures_dir / "rating_distribution.png")

    print(f"Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
