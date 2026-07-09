"""
Loads secrets from .env (TMDB_API_KEY for now). Anything that needs the key
should import get_tmdb_key() from here rather than reading os.environ
directly, so there's one place that knows where the key comes from.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_tmdb_key():
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise RuntimeError(
            "TMDB_API_KEY not set. Copy .env.example to .env and fill in your key "
            "from https://www.themoviedb.org/settings/api"
        )
    return key
