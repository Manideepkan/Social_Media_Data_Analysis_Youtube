"""
Stage 2: Exploratory Data Analysis & Visualization
Reads yt_stage1_output.csv and produces 10 plots saved to yt_eda_outputs/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "yt_stage1_output.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "yt_eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = "tab10"
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def load_data() -> pd.DataFrame:
    print(f"  Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    for col in ["published_at", "channel_published_at", "collected_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_language_distribution(df, out_dir):
    top_langs = df["language"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_langs)))
    top_langs.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Top 20 Languages — Video Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Language"); ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot1_language_dist.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")

    # Print table
    print("\n  Top 20 Languages:")
    total = len(df)
    for lang, cnt in top_langs.items():
        print(f"    {lang:<25} {cnt:>7,}  ({cnt/total*100:.1f}%)")


def plot_sentiment_distribution(df, out_dir):
    counts = df["sentiment_label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336", "#9E9E9E"], startangle=140)
    axes[0].set_title("Sentiment Distribution (Pie)")

    # Bar
    colors_map = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#9E9E9E"}
    bar_colors = [colors_map.get(l, "#999") for l in counts.index]
    counts.plot(kind="bar", ax=axes[1], color=bar_colors, edgecolor="white")
    axes[1].set_title("Sentiment Distribution (Bar)")
    axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=0)
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height()):,}",
                         (p.get_x()+p.get_width()/2, p.get_height()),
                         ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot2_sentiment_dist.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")

    print("\n  Sentiment Counts:")
    for s, c in counts.items():
        print(f"    {s:<12} {c:>7,}  ({c/len(df)*100:.1f}%)")


def plot_engagement_metrics(df, out_dir):
    metrics = ["view_count", "like_count", "comment_count", "engagement_score"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col in zip(axes.flat, metrics):
        data = df[col].dropna()
        # Log-scale histogram
        log_data = np.log1p(data)
        ax.hist(log_data, bins=60, color="#2196F3", edgecolor="white", alpha=0.8)
        ax.set_title(f"log(1+{col})", fontsize=11)
        ax.set_xlabel("log(1+value)"); ax.set_ylabel("Count")
        ax.axvline(log_data.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={data.mean():.0f}")
        ax.legend(fontsize=8)
    plt.suptitle("Engagement Metrics Distribution (log scale)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "plot3_engagement_dist.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")

    print("\n  Engagement Metric Summary:")
    for col in metrics:
        s = df[col].describe()
        print(f"    {col:<22} mean={s['mean']:>12.2f}  median={df[col].median():>12.2f}  max={s['max']:>14.0f}")


def plot_upload_hour(df, out_dir):
    hour_counts = df["upload_hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(hour_counts.index, hour_counts.values, alpha=0.4, color="#3F51B5")
    ax.plot(hour_counts.index, hour_counts.values, color="#3F51B5", linewidth=2, marker="o", markersize=4)
    ax.set_title("Upload Hour Distribution (UTC)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (0-23)"); ax.set_ylabel("Number of Videos")
    ax.set_xticks(range(0, 24))
    ax.axvline(hour_counts.idxmax(), color="red", linestyle="--", linewidth=1.5,
               label=f"Peak hour: {hour_counts.idxmax()}:00")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "plot4_upload_hour.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    print(f"\n  Peak upload hour: {hour_counts.idxmax()}:00  ({hour_counts.max():,} videos)")


def plot_upload_day(df, out_dir):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_counts = df["upload_day_of_week"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Pastel1(np.linspace(0, 1, 7))
    ax.bar([day_names[i] for i in day_counts.index], day_counts.values, color=colors, edgecolor="white")
    ax.set_title("Upload Day of Week Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Videos")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot5_upload_day.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    print("\n  Upload Day Counts:")
    for i, cnt in day_counts.items():
        print(f"    {day_names[i]:<4} {cnt:>7,}")


def plot_title_length(df, out_dir):
    top_langs = df["language"].value_counts().head(10).index.tolist()
    subset = df[df["language"].isin(top_langs)]
    fig, ax = plt.subplots(figsize=(13, 5))
    subset.boxplot(column="title_length", by="language", ax=ax,
                   patch_artist=True, notch=False, vert=True)
    ax.set_title("Title Length Distribution by Language (top 10)", fontsize=13, fontweight="bold")
    plt.suptitle("")
    ax.set_xlabel("Language"); ax.set_ylabel("Title Length (chars)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot6_title_length_by_lang.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")

    print("\n  Median Title Length by Language (top 10):")
    for lang in top_langs:
        med = df[df["language"] == lang]["title_length"].median()
        print(f"    {lang:<20} {med:.1f} chars")


def plot_correlation_heatmap(df, out_dir):
    num_cols = ["view_count", "like_count", "comment_count",
                "like_view_ratio", "comment_view_ratio", "engagement_score",
                "subscriber_count", "title_length", "title_word_count",
                "desc_word_count", "tag_count"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "plot7_correlation_heatmap.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")

    # Top correlations with view_count
    if "view_count" in corr:
        print("\n  Top correlations with view_count:")
        top_corr = corr["view_count"].drop("view_count").abs().sort_values(ascending=False).head(5)
        for col, val in top_corr.items():
            print(f"    {col:<28} r = {corr['view_count'][col]:>+.3f}")


def plot_avg_engagement_by_language(df, out_dir):
    top_langs = df["language"].value_counts().head(15).index.tolist()
    avg_eng = (df[df["language"].isin(top_langs)]
               .groupby("language")[["view_count", "like_count", "comment_count"]]
               .mean()
               .sort_values("view_count", ascending=True))
    fig, ax = plt.subplots(figsize=(10, 7))
    avg_eng.plot(kind="barh", ax=ax, width=0.7)
    ax.set_title("Avg Engagement Metrics by Language (top 15)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Count")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "plot8_avg_engagement_by_lang.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")

    print("\n  Avg View Count by Language (top 15):")
    for lang, row in avg_eng.sort_values("view_count", ascending=False).iterrows():
        print(f"    {lang:<20} views={row['view_count']:>10.0f}  likes={row['like_count']:>8.0f}")


def plot_publish_year_trend(df, out_dir):
    year_counts = df["publish_year"].value_counts().sort_index()
    year_counts = year_counts[(year_counts.index >= 2010) & (year_counts.index <= 2026)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(year_counts.index.astype(str), year_counts.values, color="#FF9800", edgecolor="white")
    ax.set_title("Videos Published per Year", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Video Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot9_publish_year.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")
    print("\n  Videos per Year:")
    for yr, cnt in year_counts.items():
        print(f"    {yr}  {cnt:>7,}")


def plot_engagement_tier_vs_sentiment(df, out_dir):
    ct = pd.crosstab(df["engagement_tier"], df["sentiment_label"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=["#4CAF50", "#F44336", "#9E9E9E"], edgecolor="white")
    ax.set_title("Sentiment Split by Engagement Tier (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Engagement Tier"); ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot10_engagement_vs_sentiment.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")

    print("\n  Engagement Tier × Sentiment (%):")
    print(ct.round(1).to_string())


def main():
    print("=" * 60)
    print("STAGE 2 — EDA & Visualization")
    print("=" * 60)

    df = load_data()

    plots = [
        ("Language Distribution",            plot_language_distribution),
        ("Sentiment Distribution",           plot_sentiment_distribution),
        ("Engagement Metrics",               plot_engagement_metrics),
        ("Upload Hour",                      plot_upload_hour),
        ("Upload Day of Week",               plot_upload_day),
        ("Title Length by Language",         plot_title_length),
        ("Correlation Heatmap",              plot_correlation_heatmap),
        ("Avg Engagement by Language",       plot_avg_engagement_by_language),
        ("Publish Year Trend",               plot_publish_year_trend),
        ("Engagement Tier vs Sentiment",     plot_engagement_tier_vs_sentiment),
    ]

    for i, (name, func) in enumerate(plots, 1):
        print(f"\n[Plot {i}/10] {name}")
        print("-" * 50)
        func(df, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"STAGE 2 COMPLETE — All 10 plots saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
