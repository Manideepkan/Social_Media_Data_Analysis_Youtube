"""
Stage 5: Dimensionality Reduction
Applies PCA, t-SNE, and UMAP to embedding matrices and saves 2D projections + plots.
Input : yt_stage4_embeddings.pkl
Output: yt_stage5_reduced.pkl  +  yt_eda_outputs/plot_reduction_*.png
"""

import os
import pickle
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "yt_stage4_embeddings.pkl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "yt_stage5_reduced.pkl")
PLOT_DIR    = os.path.join(os.path.dirname(__file__), "yt_eda_outputs")
os.makedirs(PLOT_DIR, exist_ok=True)

TSNE_SAMPLE = 5_000   # t-SNE is O(n²) — cap at 5K
UMAP_SAMPLE = 8_000
RANDOM_STATE = 42

LABEL_COLORS = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#9E9E9E"}


def load_embeddings() -> dict:
    print(f"  Loading {INPUT_PATH} ...")
    with open(INPUT_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  Sample size: {data['sample_size']:,}")
    return data


def subsample(matrix: np.ndarray, labels: np.ndarray, n: int):
    if len(matrix) > n:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(matrix), n, replace=False)
        return matrix[idx], labels[idx], idx
    return matrix, labels, np.arange(len(matrix))


def run_pca(matrix: np.ndarray, labels: np.ndarray, le, name: str) -> np.ndarray:
    t0 = time.perf_counter()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(matrix)
    elapsed = time.perf_counter() - t0

    sil = silhouette_score(coords, labels, sample_size=min(2000, len(coords)),
                           random_state=RANDOM_STATE)
    ev = pca.explained_variance_ratio_
    print(f"  [{name}] PCA done in {elapsed:.1f}s  |  "
          f"explained variance: PC1={ev[0]:.3f}  PC2={ev[1]:.3f}  |  silhouette={sil:.3f}")
    return coords, sil, elapsed


def run_tsne(matrix: np.ndarray, labels: np.ndarray, name: str) -> np.ndarray:
    sub_mat, sub_lab, _ = subsample(matrix, labels, TSNE_SAMPLE)
    t0 = time.perf_counter()
    # PCA reduce first (speeds up t-SNE significantly)
    if sub_mat.shape[1] > 50:
        pre = PCA(n_components=50, random_state=RANDOM_STATE)
        sub_mat = pre.fit_transform(sub_mat)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500,
                random_state=RANDOM_STATE, n_jobs=-1)
    coords = tsne.fit_transform(sub_mat)
    elapsed = time.perf_counter() - t0
    sil = silhouette_score(coords, sub_lab, sample_size=min(2000, len(coords)),
                           random_state=RANDOM_STATE)
    print(f"  [{name}] t-SNE done in {elapsed:.1f}s  |  "
          f"KL divergence={tsne.kl_divergence_:.3f}  |  silhouette={sil:.3f}  "
          f"(on {len(sub_lab):,} samples)")
    return coords, sub_lab, sil, elapsed


def run_umap(matrix: np.ndarray, labels: np.ndarray, name: str):
    try:
        import umap
    except ImportError:
        print(f"  [{name}] umap-learn not installed — skipping UMAP")
        return None, labels, 0.0, 0.0

    sub_mat, sub_lab, _ = subsample(matrix, labels, UMAP_SAMPLE)
    t0 = time.perf_counter()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=RANDOM_STATE)
    coords = reducer.fit_transform(sub_mat)
    elapsed = time.perf_counter() - t0
    sil = silhouette_score(coords, sub_lab, sample_size=min(2000, len(coords)),
                           random_state=RANDOM_STATE)
    print(f"  [{name}] UMAP done in {elapsed:.1f}s  |  silhouette={sil:.3f}  "
          f"(on {len(sub_lab):,} samples)")
    return coords, sub_lab, sil, elapsed


def scatter_plot(coords, labels_num, le, title: str, sil: float, elapsed: float,
                 path: str, subtitle: str = ""):
    class_names = le.classes_
    colors = [list(LABEL_COLORS.values())[i % 3] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        mask = labels_num == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[i], label=cls, alpha=0.4, s=5, linewidths=0)
    ax.set_title(f"{title}\nsilhouette={sil:.3f}  |  {elapsed:.1f}s", fontsize=12)
    if subtitle:
        ax.set_xlabel(subtitle)
    ax.legend(markerscale=4, fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def combined_plot(all_coords, all_labels, all_sils, all_times, le, method_names, path):
    n = len([c for c in all_coords if c is not None])
    if n == 0:
        return
    class_names = le.classes_
    colors = [list(LABEL_COLORS.values())[i % 3] for i in range(len(class_names))]

    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]

    idx = 0
    for coords, labels_num, sil, elapsed, name in zip(
            all_coords, all_labels, all_sils, all_times, method_names):
        if coords is None:
            continue
        ax = axes[idx]; idx += 1
        for i, cls in enumerate(class_names):
            mask = labels_num == i
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[i], label=cls, alpha=0.35, s=4, linewidths=0)
        ax.set_title(f"{name}\nsilhouette={sil:.3f} | {elapsed:.1f}s", fontsize=10)
        ax.legend(markerscale=3, fontsize=8)

    plt.suptitle("Dimensionality Reduction Comparison — YouTube Embeddings",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Combined plot saved: {path}")


def main():
    print("=" * 60)
    print("STAGE 5 — Dimensionality Reduction")
    print("=" * 60)

    data = load_embeddings()
    le   = data["label_encoder"]

    reduced: dict = {"label_encoder": le}
    all_coords, all_labels, all_sils, all_times, method_names = [], [], [], [], []

    # ── Choose embedding source (prefer Word2Vec dense, fallback TF-IDF) ──────
    if data.get("w2v_embeddings") is not None:
        base_matrix = data["w2v_embeddings"]
        base_name   = "Word2Vec"
    else:
        base_matrix = data["tfidf_matrix"].toarray()
        base_name   = "TF-IDF"

    labels = data["labels"]
    print(f"\n  Using {base_name} embeddings: {base_matrix.shape}")

    # Normalise
    base_norm = normalize(base_matrix)

    # ── PCA ───────────────────────────────────────────────────────────────────
    print("\n[1/3] PCA")
    print("-" * 40)
    pca_coords, pca_sil, pca_time = run_pca(base_norm, labels, le, base_name)
    reduced["pca_coords"] = pca_coords
    reduced["pca_labels"] = labels
    scatter_plot(pca_coords, labels, le, f"PCA — {base_name}", pca_sil, pca_time,
                 os.path.join(PLOT_DIR, "plot_reduction_pca.png"))
    all_coords.append(pca_coords); all_labels.append(labels)
    all_sils.append(pca_sil); all_times.append(pca_time); method_names.append("PCA")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print("\n[2/3] t-SNE")
    print("-" * 40)
    tsne_coords, tsne_lab, tsne_sil, tsne_time = run_tsne(base_norm, labels, base_name)
    reduced["tsne_coords"] = tsne_coords
    reduced["tsne_labels"] = tsne_lab
    scatter_plot(tsne_coords, tsne_lab, le, f"t-SNE — {base_name}", tsne_sil, tsne_time,
                 os.path.join(PLOT_DIR, "plot_reduction_tsne.png"),
                 subtitle=f"(subsampled to {len(tsne_lab):,})")
    all_coords.append(tsne_coords); all_labels.append(tsne_lab)
    all_sils.append(tsne_sil); all_times.append(tsne_time); method_names.append("t-SNE")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("\n[3/3] UMAP")
    print("-" * 40)
    umap_coords, umap_lab, umap_sil, umap_time = run_umap(base_norm, labels, base_name)
    if umap_coords is not None:
        reduced["umap_coords"] = umap_coords
        reduced["umap_labels"] = umap_lab
        scatter_plot(umap_coords, umap_lab, le, f"UMAP — {base_name}", umap_sil, umap_time,
                     os.path.join(PLOT_DIR, "plot_reduction_umap.png"),
                     subtitle=f"(subsampled to {len(umap_lab):,})")
        all_coords.append(umap_coords); all_labels.append(umap_lab)
        all_sils.append(umap_sil); all_times.append(umap_time); method_names.append("UMAP")

    # ── Combined plot ─────────────────────────────────────────────────────────
    print("\n[Combined Plot]")
    combined_plot(all_coords, all_labels, all_sils, all_times, le, method_names,
                  os.path.join(PLOT_DIR, "plot_reduction_combined.png"))

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[Save] Writing to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(reduced, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1_048_576
    print(f"       Saved ({size_mb:.1f} MB)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REDUCTION SUMMARY")
    print("=" * 60)
    print(f"\n  {'Method':<10}  {'Silhouette':>12}  {'Time (s)':>10}  {'Samples':>10}")
    print("  " + "-" * 50)
    for name, sil, t in zip(method_names, all_sils, all_times):
        n_pts = len(all_labels[method_names.index(name)])
        print(f"  {name:<10}  {sil:>12.4f}  {t:>10.1f}  {n_pts:>10,}")

    best = method_names[int(np.argmax(all_sils))]
    print(f"\n  Best separation (highest silhouette): {best}")
    print("\n[DONE] Stage 5 complete.\n")


if __name__ == "__main__":
    main()
