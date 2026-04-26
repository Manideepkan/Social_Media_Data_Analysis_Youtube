# 📊 Social Media Data Analysis — YouTube

An end-to-end NLP pipeline for large-scale YouTube video sentiment analysis, featuring multilingual text processing, multiple embedding strategies, dimensionality reduction, and GPU-accelerated classification — with an interactive analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-2.14+-green?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

This project extracts **300,000+ YouTube video records** across **13 languages** (including 8 Indian languages), processes them through a modular 8-stage NLP pipeline, and produces an interactive dashboard for real-time analytics.

### Key Highlights
- **Multilingual Support**: Hindi, Telugu, Tamil, Kannada, Bengali, Marathi, Gujarati, Urdu, Spanish, French, Korean, German, English
- **3 Embedding Strategies**: TF-IDF, Word2Vec, FastText
- **5 ML Classifiers**: Logistic Regression, LinearSVC, XGBoost, LightGBM, PyTorch MLP
- **3 Reduction Methods**: PCA, t-SNE, UMAP
- **Interactive Dashboard**: Multi-tab Plotly Dash application with real-time filtering

---

## 🔧 Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    YouTube NLP Pipeline                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5 ──►   │
│  Data         Data        EDA &       Text        Embedding     │
│  Collection   Loading     Visuals     Preproc     Generation    │
│                                                                  │
│  ──► Stage 6 ──► Stage 7 ──► Stage 8                            │
│      Dim.        Sentiment   Interactive                         │
│      Reduction   Classif.    Dashboard                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `01_data_collection.py` | YouTube API data extraction with multi-key rotation across 13 languages |
| 2 | `02_data_loading.py` | Data loading, cleaning, and feature engineering (sentiment labels, engagement tiers) |
| 3 | `03_eda_visualization.py` | Exploratory Data Analysis — 10 statistical plots (distributions, correlations, trends) |
| 4 | `04_text_preprocessing.py` | Multilingual text cleaning with language-aware stopwords and Unicode normalization |
| 5 | `05_embedding_generation.py` | TF-IDF, Word2Vec, and FastText embedding generation |
| 6 | `06_dimensionality_reduction.py` | PCA, t-SNE, and UMAP projections with silhouette evaluation |
| 7 | `07_sentiment_classification.py` | GPU-accelerated ML classification (5 models × 3 embeddings = 15 experiments) |
| 8 | `08_interactive_dashboard.py` | Multi-tab Plotly Dash dashboard with KPI cards, filters, and ML leaderboard |

---

## 📁 Repository Structure

```
Social_Media_Data_Analysis_Youtube/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── project_record.md                   # Detailed project record
│
├── 01_data_collection.py               # Stage 1: YouTube API Data Collection
├── 02_data_loading.py                  # Stage 2: Data Loading & Feature Engineering
├── 03_eda_visualization.py             # Stage 3: Exploratory Data Analysis
├── 04_text_preprocessing.py            # Stage 4: Multilingual Text Preprocessing
├── 05_embedding_generation.py          # Stage 5: Embedding Generation (TF-IDF, W2V, FT)
├── 06_dimensionality_reduction.py      # Stage 6: Dimensionality Reduction (PCA, t-SNE, UMAP)
├── 07_sentiment_classification.py      # Stage 7: ML Classification & Evaluation
├── 08_interactive_dashboard.py         # Stage 8: Interactive Analytics Dashboard
├── run_pipeline.py                     # Master Pipeline Runner (Stages 2–7)
├── pipeline_output.txt                 # Sample pipeline execution log
│
└── outputs/
    ├── eda_plots/                       # Pipeline-generated EDA & ML plots (27 images)
    ├── visualizations/                  # Seaborn statistical visualizations (10 images)
    └── extended_analysis/               # Extended analysis plots (15 images)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- YouTube Data API v3 keys ([Get yours here](https://console.cloud.google.com/apis/credentials))

### Installation

```bash
# Clone the repository
git clone https://github.com/Manideepkan/Social_Media_Data_Analysis_Youtube.git
cd Social_Media_Data_Analysis_Youtube

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your YouTube API keys as an environment variable:

```bash
# Linux / macOS
export YOUTUBE_API_KEYS="AIzaSy...,AIzaSy...,AIzaSy..."

# Windows (PowerShell)
$env:YOUTUBE_API_KEYS = "AIzaSy...,AIzaSy...,AIzaSy..."
```

### Running the Pipeline

```bash
# Run the full pipeline (Stages 2–7)
python run_pipeline.py

# Run a specific stage
python run_pipeline.py --only 3

# Resume from a specific stage
python run_pipeline.py --from 4

# Run pipeline + launch dashboard
python run_pipeline.py --dash
```

### Launch Dashboard Standalone

```bash
python 08_interactive_dashboard.py
# Open: http://127.0.0.1:8050
```

---

## 📊 Sample Outputs

### EDA Visualizations
| Plot | Description |
|------|-------------|
| Language Distribution | Top 20 languages by video count |
| Sentiment Distribution | 3-class sentiment (positive/neutral/negative) |
| Engagement Metrics | Log-scale histograms of views, likes, comments |
| Correlation Heatmap | Feature correlation matrix |
| Upload Patterns | Hour and day-of-week distributions |

### ML Classification Results
| Embedding | Best Model | F1 Score | Accuracy |
|-----------|-----------|----------|----------|
| TF-IDF | Logistic Regression | ~0.52 | ~0.52 |
| Word2Vec | XGBoost | ~0.53 | ~0.53 |
| FastText | LightGBM | ~0.52 | ~0.52 |

### Dimensionality Reduction
- **PCA**: Fast linear projection with explained variance analysis
- **t-SNE**: Non-linear embedding preserving local neighborhood structure (perplexity=30, 500 iterations)
- **UMAP**: Manifold-based reduction with cosine metric

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Data Collection** | YouTube Data API v3, urllib, isodate |
| **Data Processing** | Pandas, NumPy |
| **NLP** | Scikit-Learn (TF-IDF), Gensim (Word2Vec, FastText) |
| **ML** | Scikit-Learn, XGBoost, LightGBM, PyTorch |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Dash, Dash Bootstrap Components |
| **Reduction** | PCA, t-SNE, UMAP |

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**MANIDEEP KANDURI**

- GitHub: [@Manideepkan](https://github.com/Manideepkan)
