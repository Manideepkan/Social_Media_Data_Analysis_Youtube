"""
Stage 1: Load & Feature Engineering
Loads youtube_data1.xlsx and engineers features for the NLP pipeline.
Output: yt_stage1_output.csv
"""

import pandas as pd
import numpy as np
import os
import sys

XLSX_PATH = os.path.join(os.path.dirname(__file__), "youtube_data1.xlsx")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "yt_stage1_output.csv")

# Columns that are 100% null — drop them
ALWAYS_NULL_COLS = [
    "topic_ids", "relevant_topic_ids",
    "positive_ratio", "negative_ratio", "toxic_ratio",
    "related_video_ids",
]


def derive_sentiment(df: pd.DataFrame) -> pd.Series:
    """Derive 3-class sentiment from like_view_ratio percentiles."""
    p33 = df["like_view_ratio"].quantile(0.33)
    p66 = df["like_view_ratio"].quantile(0.66)
    conditions = [
        df["like_view_ratio"] >= p66,
        df["like_view_ratio"] < p33,
    ]
    choices = ["positive", "negative"]
    return np.select(conditions, choices, default="neutral")


def main():
    print("=" * 60)
    print("STAGE 1 — Load & Feature Engineering")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {XLSX_PATH} ...")
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    print(f"      Raw shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Drop useless columns ──────────────────────────────────────
    print("\n[2/5] Dropping 100%-null columns ...")
    to_drop = [c for c in ALWAYS_NULL_COLS if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    print(f"      Dropped    : {to_drop}")
    print(f"      Shape now  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Datetime parsing ──────────────────────────────────────────
    print("\n[3/5] Parsing datetime columns ...")
    for col in ["published_at", "channel_published_at", "collected_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    print("      Parsed: published_at, channel_published_at, collected_at")

    # ── Feature Engineering ───────────────────────────────────────
    print("\n[4/5] Engineering features ...")

    df["title_length"]      = df["title"].fillna("").str.len()
    df["title_word_count"]  = df["title"].fillna("").str.split().str.len()
    df["has_description"]   = df["description"].notna().astype(int)
    df["desc_word_count"]   = df["description"].fillna("").str.split().str.len()
    df["has_tags"]          = df["tags"].notna().astype(int)
    df["tag_count"]         = df["tags"].fillna("").str.split(",").str.len().where(df["tags"].notna(), 0)

    # Combined text for NLP
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["tags"].fillna("")
    ).str.strip()

    # Normalised language label
    df["language"] = (
        df["default_language"]
        .fillna(df.get("default_audio_language", pd.Series(dtype=str)))
        .fillna("unknown")
        .str.lower()
        .str.strip()
    )

    # Derived sentiment label
    df["sentiment_label"] = derive_sentiment(df)

    # Engagement tier
    e_p33 = df["engagement_score"].quantile(0.33)
    e_p66 = df["engagement_score"].quantile(0.66)
    df["engagement_tier"] = np.select(
        [df["engagement_score"] >= e_p66, df["engagement_score"] < e_p33],
        ["high", "low"],
        default="medium",
    )

    # Published month / year
    df["publish_year"]  = df["published_at"].dt.year
    df["publish_month"] = df["published_at"].dt.month

    print("      New columns: title_length, title_word_count, has_description,")
    print("                   desc_word_count, has_tags, tag_count, combined_text,")
    print("                   language, sentiment_label, engagement_tier,")
    print("                   publish_year, publish_month")

    # ── Save ──────────────────────────────────────────────────────
    print(f"\n[5/5] Saving to {OUTPUT_PATH} ...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"      Saved {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Summary display ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)

    print("\n--- Shape ---")
    print(f"  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

    print("\n--- Sentiment Distribution ---")
    sent_counts = df["sentiment_label"].value_counts()
    for label, cnt in sent_counts.items():
        print(f"  {label:<12}: {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    print("\n--- Language Distribution (top 15) ---")
    lang_counts = df["language"].value_counts().head(15)
    for lang, cnt in lang_counts.items():
        print(f"  {lang:<20}: {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    print("\n--- Engagement Stats ---")
    for col in ["view_count", "like_count", "comment_count", "engagement_score"]:
        s = df[col].describe()
        print(f"  {col:<22}: mean={s['mean']:>10.2f}  median={df[col].median():>10.2f}  max={s['max']:>12.0f}")

    print("\n--- Null Counts (remaining columns with nulls) ---")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    for col, cnt in nulls.items():
        print(f"  {col:<30}: {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    print("\n--- Sample Rows (first 3) ---")
    display_cols = ["video_id", "title", "language", "sentiment_label",
                    "engagement_tier", "view_count", "title_word_count"]
    sample_str = df[display_cols].head(3).to_string(index=False)
    print(sample_str.encode("ascii", errors="replace").decode("ascii"))

    print("\n[DONE] Stage 1 complete.\n")
    return df


if __name__ == "__main__":
    main()
