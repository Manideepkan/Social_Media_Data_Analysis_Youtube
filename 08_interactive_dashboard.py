"""
Stage 7 — YouTube NLP Analysis Dashboard  (Refined)
Multi-tab, fully interactive analytics dashboard with professional UI.
Run: python yt_stage7_dashboard.py  →  http://127.0.0.1:8050
"""

import os, pickle, warnings
# Limit BLAS threads BEFORE numpy is imported — prevents OOM on fragmented RAM
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

BASE          = os.path.dirname(os.path.abspath(__file__))
PARQUET       = os.path.join(BASE, "yt_dashboard_cache.parquet")
REDUCED_PATH  = os.path.join(BASE, "yt_stage5_reduced.pkl")
RESULTS_PATH  = os.path.join(BASE, "yt_stage6_results.pkl")

# ── Color System ─────────────────────────────────────────────────────────────
BRAND    = "#1565C0"
BRAND2   = "#0D47A1"
ACCENT   = "#FF6F00"
SUCCESS  = "#2E7D32"
DANGER   = "#C62828"
NEUTRAL  = "#546E7A"
BG       = "#F0F4F8"
CARD_BG  = "#FFFFFF"
TEXT     = "#1A237E"
SUBTEXT  = "#546E7A"

SENT_COLORS = {"positive": SUCCESS, "negative": DANGER, "neutral": NEUTRAL}
TIER_COLORS = {"high": BRAND, "medium": ACCENT, "low": DANGER}

LANG_PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Safe

CHART_THEME = dict(
    paper_bgcolor="white",
    plot_bgcolor="#F8FAFC",
    font=dict(family="Segoe UI, Arial, sans-serif", color="#37474F"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Data Loading ─────────────────────────────────────────────────────────────
SAMPLE_SIZE = 50_000   # keep RAM low for the dashboard

def load_data():
    import gc
    print("[Dashboard] Loading parquet cache …")
    KEEP = ["video_id","title","published_at","upload_hour","upload_day_of_week",
            "view_count","like_count","comment_count","engagement_score",
            "title_length","title_word_count","has_description","has_tags",
            "language","eff_lang","sentiment_label","engagement_tier",
            "publish_year","publish_month"]
    df = pd.read_parquet(PARQUET, columns=[c for c in KEEP
                                           if c in pd.read_parquet(PARQUET, columns=[]).columns
                                           or True])
    # keep only columns that exist
    df = df[[c for c in KEEP if c in df.columns]]

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["lang_clean"]   = df["eff_lang"].fillna(df["language"]).fillna("unknown")

    # year filter + sample
    df = df[df["publish_year"].between(2010, 2026)]
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)

    # downcast
    for col in ["upload_hour","upload_day_of_week","title_length","title_word_count",
                "has_description","has_tags","publish_year","publish_month"]:
        if col in df.columns:
            df[col] = df[col].astype("int16")
    for col in ["engagement_score"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in ["sentiment_label","engagement_tier","lang_clean","language"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df = df.reset_index(drop=True)
    gc.collect()
    print(f"  Loaded {len(df):,} rows  |  RAM: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    return df

def load_ml():
    reduced, clf = {}, {}
    try:
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "rb") as f:
                clf = pickle.load(f)
            print("  CLF results loaded.")
    except Exception as e:
        print(f"  [WARN] CLF load failed: {e}")
    return reduced, clf   # REDUCED loaded lazily in callback

_REDUCED_CACHE = {}
def get_reduced():
    if not _REDUCED_CACHE and os.path.exists(REDUCED_PATH):
        try:
            with open(REDUCED_PATH, "rb") as f:
                _REDUCED_CACHE.update(pickle.load(f))
        except Exception as e:
            print(f"  [WARN] Reduced load failed: {e}")
    return _REDUCED_CACHE

DF       = load_data()
REDUCED, CLF_RESULTS = load_ml()

TOP_LANGS    = DF["lang_clean"].value_counts().head(20).index.tolist()
SENTIMENTS   = sorted(DF["sentiment_label"].dropna().unique())
YEARS        = sorted(DF["publish_year"].dropna().unique().astype(int).tolist())
YEAR_MIN, YEAR_MAX = int(min(YEARS)), int(max(YEARS))
DAY_NAMES    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


# ── Helper ───────────────────────────────────────────────────────────────────
def apply_filters(lang, sentiments, tier, yr_range):
    mask = pd.Series(True, index=DF.index)
    if lang  != "ALL":                        mask &= DF["lang_clean"] == lang
    if sentiments and "ALL" not in sentiments: mask &= DF["sentiment_label"].isin(sentiments)
    if tier  != "ALL":                        mask &= DF["engagement_tier"] == tier
    mask &= DF["publish_year"].between(yr_range[0], yr_range[1])
    return DF[mask]

def fmt_num(n):
    if n >= 1e9:  return f"{n/1e9:.1f}B"
    if n >= 1e6:  return f"{n/1e6:.1f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return str(int(n))

def apply_theme(fig, title="", h=None):
    fig.update_layout(**CHART_THEME, title=dict(text=title, font=dict(size=14, color=TEXT), x=0))
    if h: fig.update_layout(height=h)
    return fig


# ── Card Components ───────────────────────────────────────────────────────────
def kpi_card(title, value, subtitle, icon, color, id_suffix):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Span(icon, style={"fontSize":"24px"}),
                ], style={
                    "background": f"linear-gradient(135deg,{color}22,{color}44)",
                    "border": f"1px solid {color}55",
                    "borderRadius":"12px","padding":"10px 14px",
                    "display":"inline-block","marginBottom":"8px"
                }),
                html.P(title, style={"color":SUBTEXT,"fontSize":"12px","fontWeight":"600",
                                     "textTransform":"uppercase","letterSpacing":"0.8px",
                                     "margin":"0"}),
                html.H3(value, id=f"kpi-{id_suffix}", style={"color":color,"fontWeight":"800",
                                                              "margin":"4px 0","fontSize":"26px"}),
                html.P(subtitle, style={"color":SUBTEXT,"fontSize":"11px","margin":"0"}),
            ])
        ], style={"padding":"16px"})
    ], style={
        "borderRadius":"16px","border":"none","boxShadow":"0 2px 12px rgba(0,0,0,0.08)",
        "borderTop":f"4px solid {color}","background":CARD_BG,"height":"100%"
    })

def section_header(title, subtitle=""):
    return html.Div([
        html.H5(title, style={"color":TEXT,"fontWeight":"700","margin":"0"}),
        html.P(subtitle, style={"color":SUBTEXT,"fontSize":"12px","margin":"2px 0 0"}),
    ], style={"marginBottom":"12px"})


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_language_bar(d):
    lc = d["lang_clean"].value_counts().head(15).reset_index()
    lc.columns = ["Language","Videos"]
    lc["Pct"] = (lc["Videos"] / lc["Videos"].sum() * 100).round(1)
    fig = px.bar(lc, x="Videos", y="Language", orientation="h",
                 text=lc["Videos"].apply(fmt_num),
                 color="Videos", color_continuous_scale="Blues",
                 custom_data=["Pct"])
    fig.update_traces(textposition="outside",
                      hovertemplate="<b>%{y}</b><br>Videos: %{x:,}<br>Share: %{customdata[0]}%<extra></extra>")
    fig.update_layout(**CHART_THEME, height=380,
                      title=dict(text="Top 15 Languages by Video Count",font=dict(size=14,color=TEXT),x=0),
                      coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed", title=""),
                      xaxis=dict(title="Number of Videos"))
    return fig

def fig_sentiment_donut(d):
    sc = d["sentiment_label"].value_counts().reset_index()
    sc.columns = ["Sentiment","Count"]
    colors = [SENT_COLORS.get(s, NEUTRAL) for s in sc["Sentiment"]]
    fig = go.Figure(go.Pie(
        labels=sc["Sentiment"].str.title(), values=sc["Count"],
        hole=0.55, marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>"
    ))
    total = len(d)
    fig.add_annotation(text=f"<b>{fmt_num(total)}</b><br>Videos",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color=TEXT))
    apply_theme(fig, "Sentiment Distribution", 320)
    return fig

def fig_monthly_trend(d):
    dt = d.dropna(subset=["published_at"]).copy()
    if len(dt) == 0:
        return go.Figure()
    dt["month"] = dt["published_at"].dt.to_period("M").dt.to_timestamp()
    ts = dt.groupby(["month","sentiment_label"]).size().reset_index(name="count")
    ts = ts[ts["month"].dt.year >= 2018]
    fig = px.area(ts, x="month", y="count", color="sentiment_label",
                  color_discrete_map=SENT_COLORS, line_shape="spline",
                  labels={"count":"Videos","month":"Month","sentiment_label":"Sentiment"})
    fig.update_traces(hovertemplate="%{y:,} videos<extra>%{fullData.name}</extra>")
    fig.update_layout(**CHART_THEME, height=300,
                      title=dict(text="Monthly Upload Trend by Sentiment (2018–)", font=dict(size=14,color=TEXT),x=0),
                      legend=dict(orientation="h",y=1.05,x=0),
                      xaxis_title="", yaxis_title="Videos Uploaded")
    return fig

def fig_engagement_scatter(d):
    sample = d.sample(min(8000, len(d)), random_state=42)
    sample["views_log"] = np.log10(sample["view_count"].clip(lower=1))
    sample["likes_log"] = np.log10(sample["like_count"].clip(lower=1))
    fig = px.scatter(sample, x="views_log", y="likes_log",
                     color="sentiment_label", color_discrete_map=SENT_COLORS,
                     opacity=0.45, size_max=6,
                     labels={"views_log":"Views (log₁₀)","likes_log":"Likes (log₁₀)",
                             "sentiment_label":"Sentiment"},
                     custom_data=["title","view_count","like_count","lang_clean"])
    fig.update_traces(marker=dict(size=5),
                      hovertemplate="<b>%{customdata[0]:.40s}</b><br>"
                                    "Views: %{customdata[1]:,}<br>"
                                    "Likes: %{customdata[2]:,}<br>"
                                    "Lang: %{customdata[3]}<extra></extra>")
    apply_theme(fig, "Views vs Likes (log scale) — coloured by Sentiment", 360)
    fig.update_layout(legend=dict(orientation="h", y=1.05, x=0))
    return fig

def fig_engagement_box(d):
    top_l = d["lang_clean"].value_counts().head(10).index.tolist()
    sub   = d[d["lang_clean"].isin(top_l)].copy()
    fig = px.violin(sub, x="lang_clean", y="engagement_score",
                    color="lang_clean", box=True, points=False,
                    labels={"lang_clean":"Language","engagement_score":"Engagement Score"},
                    color_discrete_sequence=LANG_PALETTE)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Engagement: %{y:.4f}<extra></extra>")
    apply_theme(fig, "Engagement Score Distribution by Language (Top 10)", 380)
    fig.update_layout(showlegend=False, xaxis_tickangle=30)
    return fig

def fig_top_videos(d):
    top = (d.nlargest(10, "view_count")
            [["title","lang_clean","view_count","like_count","comment_count","sentiment_label"]]
            .copy())
    top["View Count"]    = top["view_count"].apply(fmt_num)
    top["Like Count"]    = top["like_count"].apply(fmt_num)
    top["Comment Count"] = top["comment_count"].apply(fmt_num)
    top["Title"]         = top["title"].str[:60]
    top["Language"]      = top["lang_clean"]
    top["Sentiment"]     = top["sentiment_label"].str.title()
    return top[["Title","Language","View Count","Like Count","Comment Count","Sentiment"]]

def fig_upload_heatmap(d):
    d2 = d.copy()
    d2["day_name"] = d2["upload_day_of_week"].map({i: n for i, n in enumerate(DAY_NAMES)})
    pivot = d2.groupby(["upload_hour","day_name"]).size().reset_index(name="count")
    pivot_wide = pivot.pivot(index="upload_hour", columns="day_name", values="count").fillna(0)
    ordered_days = [x for x in DAY_NAMES if x in pivot_wide.columns]
    pivot_wide = pivot_wide[ordered_days]
    fig = go.Figure(go.Heatmap(
        z=pivot_wide.values, x=ordered_days, y=pivot_wide.index,
        colorscale="YlOrRd", hoverongaps=False,
        hovertemplate="<b>%{x} %{y}:00</b><br>Videos: %{z:,}<extra></extra>",
        colorbar=dict(title="Videos", thickness=12)
    ))
    apply_theme(fig, "Upload Activity Heatmap — Hour × Day of Week", 380)
    fig.update_layout(xaxis_title="Day of Week", yaxis_title="Hour of Day (UTC)")
    return fig

def fig_title_length_hist(d):
    fig = px.histogram(d, x="title_length", nbins=50, color="sentiment_label",
                       color_discrete_map=SENT_COLORS, barmode="overlay", opacity=0.7,
                       labels={"title_length":"Title Length (chars)","count":"Videos",
                               "sentiment_label":"Sentiment"})
    fig.update_traces(hovertemplate="Length %{x}<br>Count: %{y:,}<extra>%{fullData.name}</extra>")
    apply_theme(fig, "Title Length Distribution by Sentiment", 300)
    fig.update_layout(legend=dict(orientation="h",y=1.05,x=0))
    return fig

def fig_yearly_growth(d):
    yc = d.groupby("publish_year").size().reset_index(name="count")
    yc = yc[yc["publish_year"].between(2015, 2026)]
    yc["growth"] = yc["count"].pct_change() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=yc["publish_year"], y=yc["count"],
                         name="Videos", marker_color=BRAND,
                         hovertemplate="<b>%{x}</b><br>Videos: %{y:,}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=yc["publish_year"], y=yc["growth"],
                             name="YoY Growth %", mode="lines+markers",
                             line=dict(color=ACCENT, width=2), marker=dict(size=7),
                             hovertemplate="<b>%{x}</b><br>Growth: %{y:.1f}%<extra></extra>"), secondary_y=True)
    fig.update_layout(**CHART_THEME, height=300,
                      title=dict(text="Year-over-Year Upload Growth", font=dict(size=14,color=TEXT),x=0),
                      legend=dict(orientation="h",y=1.05,x=0),
                      xaxis=dict(dtick=1, tickangle=30))
    fig.update_yaxes(title_text="Videos Uploaded", secondary_y=False)
    fig.update_yaxes(title_text="YoY Growth %", secondary_y=True, ticksuffix="%")
    return fig

def fig_lang_sentiment_stacked(d):
    top_l = d["lang_clean"].value_counts().head(15).index.tolist()
    sub   = d[d["lang_clean"].isin(top_l)]
    ct    = pd.crosstab(sub["lang_clean"], sub["sentiment_label"], normalize="index") * 100
    ct    = ct.reindex(sub["lang_clean"].value_counts().head(15).index)
    fig   = go.Figure()
    for sent in ["positive","neutral","negative"]:
        if sent in ct.columns:
            fig.add_trace(go.Bar(
                name=sent.title(), x=ct.index, y=ct[sent].round(1),
                marker_color=SENT_COLORS[sent], opacity=0.85,
                hovertemplate="<b>%{x}</b><br>" + sent.title() + ": %{y:.1f}%<extra></extra>"
            ))
    fig.update_layout(**CHART_THEME, height=380, barmode="stack",
                      title=dict(text="Sentiment Mix by Language (Top 15, %)", font=dict(size=14,color=TEXT),x=0),
                      legend=dict(orientation="h",y=1.05,x=0),
                      xaxis_tickangle=30, yaxis=dict(ticksuffix="%"))
    return fig

def fig_avg_views_lang(d):
    top_l = d["lang_clean"].value_counts().head(15).index.tolist()
    agg   = (d[d["lang_clean"].isin(top_l)]
             .groupby("lang_clean")
             .agg(avg_views=("view_count","mean"),
                  avg_likes=("like_count","mean"),
                  video_count=("video_id","count"))
             .reset_index()
             .sort_values("avg_views", ascending=True))
    fig = px.bar(agg, x="avg_views", y="lang_clean", orientation="h",
                 color="avg_views", color_continuous_scale="Teal",
                 text=agg["avg_views"].apply(lambda v: fmt_num(int(v))),
                 custom_data=["avg_likes","video_count"],
                 labels={"avg_views":"Avg Views","lang_clean":"Language"})
    fig.update_traces(textposition="outside",
                      hovertemplate="<b>%{y}</b><br>Avg Views: %{x:,.0f}<br>"
                                    "Avg Likes: %{customdata[0]:,.0f}<br>"
                                    "Videos: %{customdata[1]:,}<extra></extra>")
    fig.update_layout(**CHART_THEME, height=420, coloraxis_showscale=False,
                      title=dict(text="Average View Count by Language (Top 15)",font=dict(size=14,color=TEXT),x=0),
                      xaxis_title="Average Views", yaxis_title="")
    return fig

def fig_lang_treemap(d):
    top_l = d["lang_clean"].value_counts().head(20)
    fig = px.treemap(
        names=top_l.index, parents=["Languages"]*len(top_l), values=top_l.values,
        color=top_l.values, color_continuous_scale="Blues",
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Videos: %{value:,}<extra></extra>",
                      textinfo="label+value")
    apply_theme(fig, "Language Distribution Treemap (Top 20)", 360)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def fig_ml_leaderboard(clf_results):
    if not clf_results or "results" not in clf_results:
        return go.Figure().update_layout(title="No ML results available")
    rows = [(emb, r) for emb, block in clf_results["results"].items() for r in block]
    rows.sort(key=lambda x: x[1]["f1"], reverse=True)
    df_m = pd.DataFrame([{
        "Rank": i+1,
        "Model": f"{emb} + {r['name']}",
        "Embedding": emb,
        "F1": round(r["f1"],4),
        "Accuracy": round(r["accuracy"],4),
        "AUC": round(r["auc"],4) if not np.isnan(r.get("auc",float("nan"))) else 0,
        "Time (s)": round(r["train_time"],1),
    } for i,(emb,r) in enumerate(rows)])
    colors = [BRAND if i==0 else (ACCENT if i==1 else "#78909C") for i in range(len(df_m))]
    fig = go.Figure(go.Bar(
        x=df_m["F1"], y=df_m["Model"], orientation="h",
        marker_color=colors, text=df_m["F1"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
        customdata=df_m[["Accuracy","AUC","Time (s)"]].values,
        hovertemplate="<b>%{y}</b><br>F1: %{x:.4f}<br>Acc: %{customdata[0]:.4f}"
                      "<br>AUC: %{customdata[1]:.4f}<br>Time: %{customdata[2]}s<extra></extra>"
    ))
    apply_theme(fig, "Model Leaderboard — Weighted F1 Score (all 15 combinations)", 500)
    fig.update_layout(xaxis=dict(title="Weighted F1 Score", range=[0,0.62]),
                      yaxis=dict(autorange="reversed", title=""),
                      showlegend=False)
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey",
                  annotation_text="0.50 baseline", annotation_position="top right")
    return fig

def fig_umap(reduced, color_by="sentiment_label"):
    coords = reduced.get("umap_coords")
    labels = reduced.get("umap_labels")
    le     = reduced.get("label_encoder")
    if coords is None:
        return go.Figure().update_layout(title="UMAP not available — run Stage 5")
    label_names = le.inverse_transform(labels)
    scatter_df  = pd.DataFrame({"x":coords[:,0],"y":coords[:,1],"label":label_names})
    if color_by == "sentiment_label":
        color_map = SENT_COLORS
    else:
        uniq   = scatter_df["label"].unique()
        color_map = {k: LANG_PALETTE[i%len(LANG_PALETTE)] for i,k in enumerate(uniq)}
    fig = px.scatter(scatter_df, x="x", y="y", color="label",
                     color_discrete_map=color_map, opacity=0.5, render_mode="webgl",
                     labels={"x":"UMAP-1","y":"UMAP-2","label":color_by.replace("_"," ").title()})
    fig.update_traces(marker=dict(size=3),
                      hovertemplate="UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<br>%{fullData.name}<extra></extra>")
    apply_theme(fig, f"UMAP 2D Projection — coloured by {color_by.replace('_',' ').title()}", 450)
    fig.update_layout(legend=dict(itemsizing="constant", orientation="v"))
    return fig

def fig_upload_day_bar(d):
    dc = d["upload_day_of_week"].value_counts().sort_index().reset_index()
    dc.columns = ["day_num","count"]
    dc["day"] = dc["day_num"].map({i:n for i,n in enumerate(DAY_NAMES)})
    colors = [BRAND if i == dc["count"].idxmax() else "#90CAF9" for i in dc.index]
    fig = go.Figure(go.Bar(x=dc["day"], y=dc["count"], marker_color=colors,
                           text=dc["count"].apply(fmt_num), textposition="outside",
                           hovertemplate="<b>%{x}</b><br>Videos: %{y:,}<extra></extra>"))
    apply_theme(fig, "Videos by Day of Week", 280)
    fig.update_layout(xaxis_title="", yaxis_title="Videos",
                      annotations=[dict(x=dc.loc[dc["count"].idxmax(),"day"],
                                        y=dc["count"].max()*1.15,
                                        text="🏆 Peak Day", showarrow=False,
                                        font=dict(color=BRAND, size=11))])
    return fig


# ── App Layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="YouTube NLP Dashboard", suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: #F0F4F8; font-family: "Segoe UI", Arial, sans-serif; }
            .tab-content { padding: 20px 0; }
            .nav-tabs .nav-link { color: #546E7A; font-weight:600; border:none; padding:10px 18px; }
            .nav-tabs .nav-link.active { color: #1565C0; border-bottom: 3px solid #1565C0; background:none; }
            .nav-tabs { border-bottom: 2px solid #E3E8EF; margin-bottom:0; }
            .chart-card { background:#fff; border-radius:16px; padding:20px;
                          box-shadow:0 2px 12px rgba(0,0,0,0.07); margin-bottom:20px; }
            .filter-label { font-size:11px; font-weight:700; text-transform:uppercase;
                            letter-spacing:0.7px; color:#546E7A; margin-bottom:4px; }
            .dashboard-header { background: linear-gradient(135deg, #1565C0 0%, #0D47A1 60%, #1A237E 100%);
                                padding: 24px 32px 20px; margin-bottom:0; }
            .filter-bar { background:#fff; padding:14px 28px; border-bottom:1px solid #E3E8EF;
                          box-shadow:0 1px 4px rgba(0,0,0,0.06); }
            .Select-control { border-radius:8px !important; }
            .dash-table-container .dash-spreadsheet-container { border-radius:8px; overflow:hidden; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

FILTER_CONTROLS = dbc.Row([
    dbc.Col([
        html.P("Language", className="filter-label"),
        dcc.Dropdown(id="f-lang",
                     options=[{"label":"🌐 All Languages","value":"ALL"}] +
                             [{"label":l.upper(),"value":l} for l in TOP_LANGS],
                     value="ALL", clearable=False,
                     style={"fontSize":"13px"}),
    ], xs=12, sm=6, md=3, lg=2),
    dbc.Col([
        html.P("Sentiment", className="filter-label"),
        dcc.Dropdown(id="f-sentiment",
                     options=[{"label":"✨ All","value":"ALL"}] +
                             [{"label":s.title(),"value":s} for s in SENTIMENTS],
                     value="ALL", clearable=False,
                     style={"fontSize":"13px"}),
    ], xs=12, sm=6, md=3, lg=2),
    dbc.Col([
        html.P("Engagement Tier", className="filter-label"),
        dcc.Dropdown(id="f-tier",
                     options=[{"label":"All Tiers","value":"ALL"},
                              {"label":"🔵 High","value":"high"},
                              {"label":"🟠 Medium","value":"medium"},
                              {"label":"🔴 Low","value":"low"}],
                     value="ALL", clearable=False,
                     style={"fontSize":"13px"}),
    ], xs=12, sm=6, md=3, lg=2),
    dbc.Col([
        html.P("Year Range", className="filter-label"),
        dcc.RangeSlider(id="f-year", min=YEAR_MIN, max=YEAR_MAX,
                        value=[2018, YEAR_MAX],
                        marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX+1, 2)},
                        tooltip={"placement":"bottom","always_visible":False},
                        step=1),
    ], xs=12, sm=12, md=3, lg=4),
    dbc.Col([
        html.P("UMAP Colour", className="filter-label"),
        dcc.Dropdown(id="f-umap-color",
                     options=[{"label":"By Sentiment","value":"sentiment_label"},
                              {"label":"By Language","value":"lang"}],
                     value="sentiment_label", clearable=False,
                     style={"fontSize":"13px"}),
    ], xs=12, sm=6, md=3, lg=2),
], className="g-3", align="center")


app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("📊 YouTube NLP Analysis Dashboard",
                            style={"color":"white","fontWeight":"800","margin":"0","fontSize":"22px"}),
                    html.P("Multilingual video analytics · 305,145 videos · GPU-accelerated ML insights",
                           style={"color":"#90CAF9","fontSize":"12px","margin":"4px 0 0"}),
                ], md=8),
                dbc.Col([
                    html.Div([
                        html.Span("🟢 Live", style={"color":"#A5D6A7","fontSize":"12px","fontWeight":"600"}),
                        html.Span(" · Data: youtube_data1.xlsx",
                                  style={"color":"#B0BEC5","fontSize":"11px"}),
                    ], style={"textAlign":"right","paddingTop":"8px"})
                ], md=4),
            ])
        ], fluid=True)
    ], className="dashboard-header"),

    # ── Filters ───────────────────────────────────────────────────────────────
    html.Div([
        dbc.Container(FILTER_CONTROLS, fluid=True)
    ], className="filter-bar"),

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    dbc.Container([
        dbc.Row(id="kpi-row", className="g-3 mt-1"),

        # ── Tabs ──────────────────────────────────────────────────────────────
        dbc.Tabs(id="main-tabs", active_tab="tab-overview", className="mt-3", children=[

            # ── Overview Tab ──────────────────────────────────────────────────
            dbc.Tab(tab_id="tab-overview", label="📈 Overview", children=[
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-lang-bar"), className="chart-card"), md=7),
                    dbc.Col(html.Div(dcc.Graph(id="g-sentiment-donut"), className="chart-card"), md=5),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-monthly-trend"), className="chart-card"), md=8),
                    dbc.Col(html.Div(dcc.Graph(id="g-upload-day"), className="chart-card"), md=4),
                ]),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-yearly-growth"), className="chart-card"), md=12),
                ]),
            ]),

            # ── Engagement Tab ────────────────────────────────────────────────
            dbc.Tab(tab_id="tab-engagement", label="🔥 Engagement", children=[
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-scatter"), className="chart-card"), md=7),
                    dbc.Col(html.Div(dcc.Graph(id="g-eng-box"), className="chart-card"), md=5),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            section_header("🏆 Top 10 Most-Viewed Videos",
                                           "Filtered by current selection"),
                            dash_table.DataTable(
                                id="top-videos-table",
                                style_table={"overflowX":"auto"},
                                style_header={"backgroundColor":BRAND,"color":"white",
                                              "fontWeight":"bold","fontSize":"12px","border":"none"},
                                style_cell={"fontSize":"12px","padding":"8px 12px",
                                            "border":"1px solid #E8EDF2","textAlign":"left",
                                            "maxWidth":"300px","overflow":"hidden",
                                            "textOverflow":"ellipsis"},
                                style_data_conditional=[
                                    {"if":{"row_index":"odd"},"backgroundColor":"#F8FAFC"},
                                    {"if":{"column_id":"Sentiment","filter_query":'{Sentiment} = "Positive"'},
                                     "color":SUCCESS,"fontWeight":"bold"},
                                    {"if":{"column_id":"Sentiment","filter_query":'{Sentiment} = "Negative"'},
                                     "color":DANGER,"fontWeight":"bold"},
                                ],
                                tooltip_delay=0, tooltip_duration=None,
                            )
                        ], className="chart-card")
                    ], md=12),
                ]),
            ]),

            # ── Content Patterns Tab ──────────────────────────────────────────
            dbc.Tab(tab_id="tab-content", label="📝 Content Patterns", children=[
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-upload-heatmap"), className="chart-card"), md=7),
                    dbc.Col(html.Div(dcc.Graph(id="g-title-hist"), className="chart-card"), md=5),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            section_header("📊 Content Feature Summary"),
                            html.Div(id="content-stats-table")
                        ], className="chart-card")
                    ], md=12),
                ]),
            ]),

            # ── Language Intelligence Tab ──────────────────────────────────────
            dbc.Tab(tab_id="tab-language", label="🌐 Language Intelligence", children=[
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-lang-treemap"), className="chart-card"), md=5),
                    dbc.Col(html.Div(dcc.Graph(id="g-lang-sentiment"), className="chart-card"), md=7),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-avg-views-lang"), className="chart-card"), md=12),
                ]),
            ]),

            # ── ML Insights Tab ───────────────────────────────────────────────
            dbc.Tab(tab_id="tab-ml", label="🤖 ML Insights", children=[
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id="g-ml-leaderboard"), className="chart-card"), md=7),
                    dbc.Col([
                        html.Div([
                            section_header("📐 Model Key Metrics",
                                           "Best model per embedding type"),
                            html.Div(id="ml-best-table"),
                        ], className="chart-card"),
                    ], md=5),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(html.Div([
                        section_header("🗺️ UMAP 2D Projection",
                                       "Word2Vec embeddings reduced to 2D"),
                        dcc.Graph(id="g-umap"),
                    ], className="chart-card"), md=12),
                ]),
            ]),
        ]),
    ], fluid=True, style={"paddingBottom":"40px"}),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row","children"),
    Output("g-lang-bar","figure"),
    Output("g-sentiment-donut","figure"),
    Output("g-monthly-trend","figure"),
    Output("g-upload-day","figure"),
    Output("g-yearly-growth","figure"),
    Output("g-scatter","figure"),
    Output("g-eng-box","figure"),
    Output("top-videos-table","data"),
    Output("top-videos-table","columns"),
    Output("top-videos-table","tooltip_data"),
    Output("g-upload-heatmap","figure"),
    Output("g-title-hist","figure"),
    Output("content-stats-table","children"),
    Output("g-lang-treemap","figure"),
    Output("g-lang-sentiment","figure"),
    Output("g-avg-views-lang","figure"),
    Input("f-lang","value"),
    Input("f-sentiment","value"),
    Input("f-tier","value"),
    Input("f-year","value"),
)
def update_all(lang, sentiment, tier, yr_range):
    d = apply_filters(lang, [sentiment], tier, yr_range)

    if len(d) == 0:
        empty = go.Figure().update_layout(title="No data for selection")
        kpis  = dbc.Row(dbc.Col(dbc.Alert("No data matches the current filters.",color="warning")))
        return (kpis, empty, empty, empty, empty, empty,
                empty, empty, [], [], [], empty, empty,
                html.P("No data"), empty, empty, empty)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    pos_pct = (d["sentiment_label"]=="positive").mean()*100
    neg_pct = (d["sentiment_label"]=="negative").mean()*100
    kpis = dbc.Row([
        dbc.Col(kpi_card("Total Videos",    fmt_num(len(d)),
                         f"{d['lang_clean'].nunique()} languages", "🎬", BRAND, "total"), md=2),
        dbc.Col(kpi_card("Total Views",     fmt_num(d["view_count"].sum()),
                         f"avg {fmt_num(int(d['view_count'].mean()))} / video", "👁️", "#0097A7", "views"), md=2),
        dbc.Col(kpi_card("Total Likes",     fmt_num(d["like_count"].sum()),
                         f"avg {fmt_num(int(d['like_count'].mean()))} / video", "👍", ACCENT, "likes"), md=2),
        dbc.Col(kpi_card("Avg Engagement",  f"{d['engagement_score'].mean():.4f}",
                         f"median: {d['engagement_score'].median():.4f}", "📊", "#6A1B9A", "eng"), md=2),
        dbc.Col(kpi_card("Positive %",      f"{pos_pct:.1f}%",
                         f"negative: {neg_pct:.1f}%", "😊", SUCCESS, "pos"), md=2),
        dbc.Col(kpi_card("Has Description", f"{d['has_description'].mean()*100:.1f}%",
                         f"has tags: {d['has_tags'].mean()*100:.1f}%", "📄", "#00796B", "desc"), md=2),
    ], className="g-3")

    # ── Charts ────────────────────────────────────────────────────────────────
    top_df      = fig_top_videos(d)
    tbl_cols    = [{"name":c,"id":c} for c in top_df.columns]
    tbl_data    = top_df.to_dict("records")
    tbl_tooltip = [{c: {"value":row.get("Title",""),"type":"markdown"}
                    for c in top_df.columns} for row in tbl_data]

    # Content stats table
    stats = {
        "Metric": ["Total Videos","Avg Title Length","Avg Word Count","Has Description",
                   "Has Tags","Avg Views","Avg Likes","Avg Comments"],
        "Value":  [
            f"{len(d):,}",
            f"{d['title_length'].mean():.1f} chars",
            f"{d['title_word_count'].mean():.1f} words",
            f"{d['has_description'].mean()*100:.1f}%",
            f"{d['has_tags'].mean()*100:.1f}%",
            fmt_num(int(d['view_count'].mean())),
            fmt_num(int(d['like_count'].mean())),
            fmt_num(int(d['comment_count'].mean())),
        ]
    }
    stats_tbl = dash_table.DataTable(
        data=pd.DataFrame(stats).to_dict("records"),
        columns=[{"name":"Metric","id":"Metric"},{"name":"Value","id":"Value"}],
        style_header={"backgroundColor":BRAND,"color":"white","fontWeight":"bold","fontSize":"12px"},
        style_cell={"fontSize":"13px","padding":"8px 14px","border":"1px solid #E8EDF2"},
        style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#F8FAFC"}],
    )

    return (
        kpis,
        fig_language_bar(d),
        fig_sentiment_donut(d),
        fig_monthly_trend(d),
        fig_upload_day_bar(d),
        fig_yearly_growth(d),
        fig_engagement_scatter(d),
        fig_engagement_box(d),
        tbl_data, tbl_cols, tbl_tooltip,
        fig_upload_heatmap(d),
        fig_title_length_hist(d),
        stats_tbl,
        fig_lang_treemap(d),
        fig_lang_sentiment_stacked(d),
        fig_avg_views_lang(d),
    )


@app.callback(
    Output("g-ml-leaderboard","figure"),
    Output("ml-best-table","children"),
    Output("g-umap","figure"),
    Input("f-umap-color","value"),
    Input("main-tabs","active_tab"),
)
def update_ml_tab(umap_color, active_tab):
    if active_tab != "tab-ml":
        raise dash.exceptions.PreventUpdate

    fig_leader = fig_ml_leaderboard(CLF_RESULTS)

    # Best-per-embedding mini table
    best_rows = []
    if CLF_RESULTS and "results" in CLF_RESULTS:
        for emb, block in CLF_RESULTS["results"].items():
            best = max(block, key=lambda r: r["f1"])
            auc  = f"{best['auc']:.4f}" if not np.isnan(best.get("auc",float("nan"))) else "N/A"
            best_rows.append({
                "Embedding": emb,
                "Best Model": best["name"],
                "F1": f"{best['f1']:.4f}",
                "Accuracy": f"{best['accuracy']:.4f}",
                "AUC-ROC": auc,
                "GPU": "✓" if best["name"] in ("XGBoost","LightGBM","PyTorch MLP") else "–",
            })
    best_tbl = dash_table.DataTable(
        data=best_rows,
        columns=[{"name":c,"id":c} for c in (best_rows[0].keys() if best_rows else [])],
        style_header={"backgroundColor":BRAND,"color":"white","fontWeight":"bold","fontSize":"12px"},
        style_cell={"fontSize":"12px","padding":"8px 12px","border":"1px solid #E8EDF2"},
        style_data_conditional=[
            {"if":{"row_index":"odd"},"backgroundColor":"#F8FAFC"},
            {"if":{"column_id":"GPU","filter_query":'{GPU} = "✓"'},
             "color":SUCCESS,"fontWeight":"bold"},
        ],
    ) if best_rows else html.P("No ML results available.")

    return fig_leader, best_tbl, fig_umap(get_reduced(), umap_color)


# ── Run ───────────────────────────────────────────────────────────────────────
def main():
    print("\n[Dashboard] Starting at http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, host="127.0.0.1", port=8050)

if __name__ == "__main__":
    main()
