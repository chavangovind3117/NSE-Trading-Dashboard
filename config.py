import os
from pathlib import Path

try:
    import streamlit as st
    _has_streamlit = True
except ImportError:
    _has_streamlit = False


def _load_dotenv(dotenv_path=None):
    path = Path(dotenv_path or Path(__file__).parent / ".env")
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _get_secret(key, default=""):
    if _has_streamlit and hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)


TELEGRAM_TOKEN = _get_secret("TELEGRAM_TOKEN", "")
CHAT_ID = _get_secret("CHAT_ID", "")
GROQ_API_KEY = _get_secret("GROQ_API_KEY", "")
GROQ_MODEL = _get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
