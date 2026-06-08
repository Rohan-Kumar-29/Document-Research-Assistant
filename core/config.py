import os


def get_api_key() -> str:
    """
    Return the Google API key, looking in both places it can live:

    - Locally: a .env file loaded into os.environ (via python-dotenv).
    - On Streamlit Community Cloud: st.secrets, which does NOT populate
      os.environ automatically.

    Checking both means the same code runs in development and in deployment.
    """
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    # Fall back to Streamlit secrets if available (only when running under
    # Streamlit). Imported lazily so the core modules don't hard-depend on it.
    try:
        import streamlit as st

        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    raise RuntimeError(
        "GOOGLE_API_KEY not found. Set it in a local .env file, or in "
        "Streamlit Cloud under Manage app → Settings → Secrets."
    )
