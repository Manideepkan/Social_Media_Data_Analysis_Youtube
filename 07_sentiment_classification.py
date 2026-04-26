"""
Stage 6: GPU-Accelerated ML Classification & Evaluation
Classifiers: Logistic Regression, LinearSVC, XGBoost (GPU), LightGBM (GPU), PyTorch MLP (GPU)
Input : yt_stage4_embeddings.pkl
Output: yt_stage6_results.pkl  +  yt_eda_outputs/plot_classification_*.png
"""

import os
import pickle
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
)

warnings.filterwarnings("ignore")

INPUT_PATH   = os.path.join(os.path.dirname(__file__), "yt_stage4_embeddings.pkl")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "yt_stage6_results.pkl")
PLOT_DIR     = os.path.join(os.path.dirname(__file__), "yt_eda_outputs")
os.makedirs(PLOT_DIR, exist_ok=True)

TEST_SIZE    = 0.20
RANDOM_STATE = 42


# ── GPU / device detection ────────────────────────────────────────────────────
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [GPU] {name}  ({vram:.1f} GB VRAM)")
            return "cuda"
        print("  [GPU] CUDA not available — using CPU")
    except ImportError:
        print("  [GPU] PyTorch not installed — using CPU")
    return "cpu"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_embeddings() -> dict:
    print(f"  Loading {INPUT_PATH} ...")
    with open(INPUT_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  Sample size: {data['sample_size']:,}")
    return data


def get_matrix(data: dict, key: str) -> np.ndarray | None:
    mat = data.get(key)
    if mat is None:
        return None
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return normalize(mat.astype(np.float32))


# ── Evaluation helper ─────────────────────────────────────────────────────────
def compute_metrics(name, y_test, y_pred, y_proba, le, train_time) -> dict:
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    rep  = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    auc = float("nan")
    if y_proba is not None:
        try:
            if len(le.classes_) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            pass

    return dict(
        name=name, accuracy=acc, f1=f1, precision=prec, recall=rec,
        auc=auc, confusion_matrix=cm, report=rep, train_time=train_time,
        y_pred=y_pred, y_test=y_test,
    )


def print_metrics(res: dict):
    auc_str = f"{res['auc']:.4f}" if not np.isnan(res['auc']) else "N/A"
    print(f"\n  Model         : {res['name']}")
    print(f"  Train time    : {res['train_time']:.1f}s")
    print(f"  Accuracy      : {res['accuracy']:.4f}")
    print(f"  Weighted F1   : {res['f1']:.4f}")
    print(f"  Precision     : {res['precision']:.4f}")
    print(f"  Recall        : {res['recall']:.4f}")
    print(f"  AUC-ROC       : {auc_str}")
    print(f"\n  Classification Report:")
    for line in res["report"].splitlines():
        print(f"    {line}")


# ── Classifiers ───────────────────────────────────────────────────────────────

def run_logistic(X_tr, X_te, y_tr, y_te, le):
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                             n_jobs=-1, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    return compute_metrics("Logistic Regression", y_te, y_pred, y_proba, le, elapsed)


def run_svm(X_tr, X_te, y_tr, y_te, le):
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    y_pred = clf.predict(X_te)
    scores = clf.decision_function(X_te)
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    y_proba = e / e.sum(axis=1, keepdims=True)
    return compute_metrics("Linear SVM", y_te, y_pred, y_proba, le, elapsed)


def run_xgboost(X_tr, X_te, y_tr, y_te, le, device):
    try:
        import xgboost as xgb
    except ImportError:
        print("  [SKIP] xgboost not installed")
        return None

    # Try GPU first, fall back to CPU
    dev = "cuda" if device == "cuda" else "cpu"
    params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
        device=dev, tree_method="hist",
        n_jobs=-1,
    )
    try:
        clf = xgb.XGBClassifier(**params)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        elapsed = time.perf_counter() - t0
        device_used = dev.upper()
    except Exception:
        params["device"] = "cpu"
        clf = xgb.XGBClassifier(**params)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        elapsed = time.perf_counter() - t0
        device_used = "CPU"

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    print(f"    (ran on {device_used})")
    return compute_metrics("XGBoost", y_te, y_pred, y_proba, le, elapsed)


def run_lightgbm(X_tr, X_te, y_tr, y_te, le, device):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [SKIP] lightgbm not installed")
        return None

    dev = "gpu" if device == "cuda" else "cpu"
    params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        device=dev, random_state=RANDOM_STATE, n_jobs=-1,
        verbose=-1,
    )
    try:
        clf = lgb.LGBMClassifier(**params)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0
        device_used = dev.upper()
    except Exception:
        params["device"] = "cpu"
        clf = lgb.LGBMClassifier(**params)
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0
        device_used = "CPU"

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    print(f"    (ran on {device_used})")
    return compute_metrics("LightGBM", y_te, y_pred, y_proba, le, elapsed)


def run_pytorch_mlp(X_tr, X_te, y_tr, y_te, le, device):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  [SKIP] PyTorch not installed")
        return None

    n_classes = len(le.classes_)
    in_dim    = X_tr.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128),    nn.ReLU(),
                nn.Linear(128, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dev_str = str(dev).upper()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    ds     = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=512, shuffle=True, pin_memory=(device == "cuda"))

    model     = MLP().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, steps_per_epoch=len(loader), epochs=20
    )

    t0 = time.perf_counter()
    model.train()
    for epoch in range(20):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

    elapsed = time.perf_counter() - t0
    print(f"    (ran on {dev_str}, 20 epochs, {elapsed:.1f}s)")

    model.eval()
    with torch.no_grad():
        logits  = model(X_te_t.to(dev)).cpu()
        y_proba = torch.softmax(logits, dim=1).numpy()
        y_pred  = y_proba.argmax(axis=1)

    return compute_metrics("PyTorch MLP", y_te, y_pred, y_proba, le, elapsed)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_metrics_comparison(results: list, emb_name: str):
    results = [r for r in results if r]
    if not results:
        return
    names   = [r["name"] for r in results]
    metrics = ["accuracy", "f1", "precision", "recall"]
    labels  = ["Accuracy", "F1", "Precision", "Recall"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    x = np.arange(len(names))
    w = 0.18

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [r[m] for r in results]
        bars = ax.bar(x + i * w, vals, w, label=lbl, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Classifier Comparison — {emb_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"plot_clf_{emb_name.lower().replace(' ','_')}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  Saved: {path}")


def plot_best_confusion(results: list, emb_name: str, le):
    results = [r for r in results if r]
    if not results:
        return
    best = max(results, key=lambda r: r["f1"])
    cm   = best["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix — {best['name']} ({emb_name})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"plot_cm_{emb_name.lower().replace(' ','_')}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  Saved: {path}")


# ── Per-embedding block ───────────────────────────────────────────────────────

def run_block(data, matrix_key, emb_name, device):
    matrix = get_matrix(data, matrix_key)
    if matrix is None:
        print(f"  [SKIP] {emb_name} not found")
        return []

    labels = data["labels"]
    le     = data["label_encoder"]
    print(f"\n  Embedding : {emb_name}  {matrix.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        matrix, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    print(f"  Train: {X_tr.shape[0]:,}   Test: {X_te.shape[0]:,}")

    results = []
    for label, fn, kwargs in [
        ("Logistic Regression", run_logistic, {}),
        ("Linear SVM",          run_svm,      {}),
        ("XGBoost",             run_xgboost,  {"device": device}),
        ("LightGBM",            run_lightgbm, {"device": device}),
        ("PyTorch MLP",         run_pytorch_mlp, {"device": device}),
    ]:
        print(f"\n  --- {label} ---")
        res = fn(X_tr, X_te, y_tr, y_te, le, **kwargs)
        if res:
            print_metrics(res)
            results.append(res)

    plot_metrics_comparison(results, emb_name)
    plot_best_confusion(results, emb_name, le)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STAGE 6 — GPU-Accelerated Classification")
    print("=" * 60)

    device = get_device()
    data   = load_embeddings()
    le     = data["label_encoder"]

    all_results = {}

    for matrix_key, emb_name in [
        ("tfidf_matrix",  "TF-IDF"),
        ("w2v_embeddings","Word2Vec"),
        ("ft_embeddings", "FastText"),
    ]:
        print(f"\n{'='*60}\nBLOCK — {emb_name}\n{'='*60}")
        block = run_block(data, matrix_key, emb_name, device)
        all_results[emb_name] = block

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    all_flat = [(emb, r) for emb, block in all_results.items() for r in block]
    if all_flat:
        print(f"\n  {'Embedding':<12}  {'Model':<22}  {'Acc':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'AUC':>7}  {'Time':>6}")
        print("  " + "-" * 88)
        for emb, r in sorted(all_flat, key=lambda x: x[1]["f1"], reverse=True):
            auc_s = f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "   N/A"
            print(f"  {emb:<12}  {r['name']:<22}  {r['accuracy']:.4f}  "
                  f"{r['f1']:.4f}  {r['precision']:.4f}  {r['recall']:.4f}  "
                  f"{auc_s}  {r['train_time']:>5.1f}s")

        best_emb, best_r = max(all_flat, key=lambda x: x[1]["f1"])
        print(f"\n  BEST: {best_r['name']} on {best_emb}  (F1={best_r['f1']:.4f})")

    # ── Save ──────────────────────────────────────────────────────
    print(f"\n[Save] {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"results": all_results, "label_encoder": le}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"       {os.path.getsize(OUTPUT_PATH)/1_048_576:.1f} MB saved")
    print("\n[DONE] Stage 6 complete.\n")


if __name__ == "__main__":
    main()
