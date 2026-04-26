"""
Stage 4: Embedding Generation
Produces TF-IDF, Word2Vec, and FastText embeddings on a sampled subset.
Input : yt_stage3_output.csv
Output: yt_stage4_embeddings.pkl
"""

import os
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "yt_stage3_output.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "yt_stage4_embeddings.pkl")

SAMPLE_SIZE     = 20_000   # rows used for dense embeddings
TFIDF_MAX_FEAT  = 8_000
W2V_SIZE        = 128
W2V_WINDOW      = 5
W2V_EPOCHS      = 10
RANDOM_STATE    = 42


def load_data() -> pd.DataFrame:
    print(f"  Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    # Keep only rows that have actual text
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    print(f"  {df.shape[0]:,} usable rows")
    return df


def sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) > n:
        sampled = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  Sampled {n:,} rows from {len(df):,}")
    else:
        sampled = df.reset_index(drop=True)
        print(f"  Using all {len(df):,} rows (below sample threshold)")
    return sampled


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def build_tfidf(texts: list[str]) -> tuple:
    print(f"  Fitting TF-IDF (max_features={TFIDF_MAX_FEAT}, bigrams) ...")
    t0 = time.perf_counter()
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEAT,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vec.fit_transform(texts)
    elapsed = time.perf_counter() - t0
    print(f"  TF-IDF matrix: {matrix.shape}  |  {elapsed:.1f}s")
    return vec, matrix


# ── Word2Vec ──────────────────────────────────────────────────────────────────

def build_word2vec(tokenized: list[list[str]]) -> tuple:
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  [SKIP] gensim not installed — skipping Word2Vec")
        return None, None

    print(f"  Training Word2Vec (dim={W2V_SIZE}, window={W2V_WINDOW}, epochs={W2V_EPOCHS}) ...")
    t0 = time.perf_counter()
    model = Word2Vec(
        sentences=tokenized,
        vector_size=W2V_SIZE,
        window=W2V_WINDOW,
        min_count=1,
        workers=4,
        epochs=W2V_EPOCHS,
        seed=RANDOM_STATE,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Word2Vec trained in {elapsed:.1f}s  |  vocab={len(model.wv):,}")

    def mean_pool(tokens: list[str]) -> np.ndarray:
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(W2V_SIZE)

    embeddings = np.vstack([mean_pool(toks) for toks in tokenized])
    print(f"  Word2Vec embeddings: {embeddings.shape}")
    return model, embeddings


# ── FastText ──────────────────────────────────────────────────────────────────

def build_fasttext(tokenized: list[list[str]]) -> tuple:
    try:
        from gensim.models import FastText
    except ImportError:
        print("  [SKIP] gensim not installed — skipping FastText")
        return None, None

    print(f"  Training FastText (dim={W2V_SIZE}, window={W2V_WINDOW}, epochs={W2V_EPOCHS}) ...")
    t0 = time.perf_counter()
    model = FastText(
        sentences=tokenized,
        vector_size=W2V_SIZE,
        window=W2V_WINDOW,
        min_count=1,
        workers=4,
        epochs=W2V_EPOCHS,
        seed=RANDOM_STATE,
    )
    elapsed = time.perf_counter() - t0
    print(f"  FastText trained in {elapsed:.1f}s  |  vocab={len(model.wv):,}")

    def mean_pool(tokens: list[str]) -> np.ndarray:
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(W2V_SIZE)

    embeddings = np.vstack([mean_pool(toks) for toks in tokenized])
    print(f"  FastText embeddings: {embeddings.shape}")
    return model, embeddings


# ── Label Encoder ─────────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame) -> np.ndarray:
    le = LabelEncoder()
    labels = le.fit_transform(df["sentiment_label"].fillna("neutral"))
    print(f"  Labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return labels, le


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STAGE 4 — Embedding Generation")
    print("=" * 60)

    df_full = load_data()
    df = sample_df(df_full, SAMPLE_SIZE)

    texts     = df["clean_text"].tolist()
    tokenized = [t.split() for t in texts]

    results: dict = {"sample_df": df, "sample_size": len(df)}

    # TF-IDF
    print("\n[1/3] TF-IDF")
    print("-" * 40)
    tfidf_vec, tfidf_matrix = build_tfidf(texts)
    results["tfidf_vectorizer"] = tfidf_vec
    results["tfidf_matrix"]     = tfidf_matrix

    # Word2Vec
    print("\n[2/3] Word2Vec")
    print("-" * 40)
    w2v_model, w2v_embeddings = build_word2vec(tokenized)
    results["w2v_model"]      = w2v_model
    results["w2v_embeddings"] = w2v_embeddings

    # FastText
    print("\n[3/3] FastText")
    print("-" * 40)
    ft_model, ft_embeddings = build_fasttext(tokenized)
    results["ft_model"]      = ft_model
    results["ft_embeddings"] = ft_embeddings

    # Labels
    print("\n[Labels]")
    labels, le = encode_labels(df)
    results["labels"]         = labels
    results["label_encoder"]  = le

    # Save
    print(f"\n[Save] Writing to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1_048_576
    print(f"       Saved ({size_mb:.1f} MB)")

    # Summary
    print("\n" + "=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)
    print(f"\n  Sample size         : {len(df):,}")
    print(f"  TF-IDF shape        : {tfidf_matrix.shape}  (sparse)")
    if w2v_embeddings is not None:
        print(f"  Word2Vec shape      : {w2v_embeddings.shape}  (dense)")
    if ft_embeddings is not None:
        print(f"  FastText shape      : {ft_embeddings.shape}  (dense)")
    print(f"  Label classes       : {list(le.classes_)}")
    print(f"  Label distribution  :")
    unique, cnts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, cnts):
        print(f"    {le.inverse_transform([u])[0]:<12} {c:>6,}  ({c/len(labels)*100:.1f}%)")

    if w2v_embeddings is not None:
        print(f"\n  Word2Vec — top 10 most similar to 'music':")
        if "music" in w2v_model.wv:
            for word, score in w2v_model.wv.most_similar("music", topn=10):
                print(f"    {word:<20} {score:.4f}")
        else:
            print("    ('music' not in vocabulary — try another seed word)")

    print("\n[DONE] Stage 4 complete.\n")


if __name__ == "__main__":
    main()
