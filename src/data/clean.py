import html
import re
import unicodedata

import pandas as pd


def repair_mojibake(text):
    """
    Repair common UTF-8 text that was incorrectly decoded as Latin-1.
    """
    if not isinstance(text, str):
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_text(text):
    """
    Lightweight text normalization for support messages.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Repair common encoding artifacts.
    text = repair_mojibake(text)

    # Decode HTML entities such as &amp;.
    text = html.unescape(text)

    # Normalize Unicode representations.
    text = unicodedata.normalize("NFKC", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataframe(df):
    """
    Clean the dataframe while preserving raw fields.

    Timestamp parsing is intentionally deferred. The raw timestamp
    is preserved and can be parsed only for the subset of data
    needed by downstream conversation reconstruction.
    """
    out = df.copy()

    # Preserve original values for audit/debugging.
    if "text" in out.columns:
        out["raw_text"] = out["text"]

    if "created_at" in out.columns:
        out["raw_created_at"] = out["created_at"]

    # Clean text.
    if "text" in out.columns:
        out["text"] = out["text"].map(clean_text)

    # Do NOT parse 1M timestamps here.
    # Keep created_at as the original string.
    return out


def cleaning_quality_report(df):
    """
    Return basic cleaning-quality checks.
    """
    return {
        "rows": len(df),
        "missing_text": int(df["text"].isna().sum()),
        "empty_text": int((df["text"].str.strip() == "").sum()),
        "unique_tweet_ids": int(df["tweet_id"].nunique()),
    }