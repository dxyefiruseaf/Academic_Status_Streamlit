"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Academic Status Predictor  —  Streamlit Dashboard                 ║
║  Model : HistGradientBoosting Ensemble  |  CV Macro F1 ≈ 0.86              ║
║  Classes: 0 = Đạt (Pass)  |  1 = Cảnh báo (Warning)  |  2 = Thôi học      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import io
import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.sparse as sp
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Status Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — light, professional dashboard style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #F4F6FB;
    font-family: "Inter", "Segoe UI", sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A8A 0%, #1e40af 100%);
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.2); }
[data-testid="stSidebar"] .sidebar-metric {
    background: rgba(255,255,255,.1);
    border-radius: 10px; padding: 10px 14px; margin: 6px 0;
    font-size: .84rem;
}
[data-testid="stSidebar"] .sidebar-metric strong { font-size: 1.1rem; }

/* ── Top banner ───────────────────────────────────────────────────────────── */
.top-banner {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #38BDF8 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: 0 4px 24px rgba(37,99,235,.25);
}
.top-banner h1 { margin: 0; color: #fff; font-size: 1.9rem; font-weight: 800; }
.top-banner p  { margin: 4px 0 0; color: rgba(255,255,255,.82); font-size: .95rem; }

/* ── White card ───────────────────────────────────────────────────────────── */
.card {
    background: #FFFFFF; border-radius: 14px; padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,.07); margin-bottom: 16px;
}
.card-title {
    font-size: .75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.3px; color: #64748B; margin-bottom: 10px;
}

/* ── Metric tile grid ─────────────────────────────────────────────────────── */
.metric-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.metric-tile {
    flex: 1; min-width: 120px; background: #fff; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,.07);
    border-top: 4px solid var(--accent); text-align: center;
}
.metric-tile .val { font-size: 1.85rem; font-weight: 800; color: var(--accent); line-height: 1.1; }
.metric-tile .lbl { font-size: .7rem; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 1px; color: #94A3B8; margin-top: 5px; }

/* ── Status badges ────────────────────────────────────────────────────────── */
.badge { display: inline-flex; align-items: center; gap: 6px;
         border-radius: 999px; padding: 7px 20px;
         font-size: .95rem; font-weight: 700; }
.badge-0 { background: #D1FAE5; color: #065F46; }
.badge-1 { background: #FEF3C7; color: #92400E; }
.badge-2 { background: #FEE2E2; color: #991B1B; }

/* ── Probability progress bars ────────────────────────────────────────────── */
.prob-row { margin: 10px 0; }
.prob-label { display: flex; justify-content: space-between;
              font-size: .83rem; font-weight: 600; color: #334155; margin-bottom: 5px; }
.prob-bar-bg { background: #E2E8F0; border-radius: 999px; height: 12px; overflow: hidden; }
.prob-bar-fill { height: 12px; border-radius: 999px; }

/* ── Student info table ───────────────────────────────────────────────────── */
.info-table { width: 100%; border-collapse: collapse; }
.info-table td { padding: 8px 10px; font-size: .84rem; border-bottom: 1px solid #F1F5F9; }
.info-table td:first-child { font-weight: 600; color: #64748B; width: 48%; white-space: nowrap; }
.info-table td:last-child  { color: #1E293B; }

/* ── Attendance heatmap grid ──────────────────────────────────────────────── */
.att-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(50px, 1fr));
            gap: 6px; margin-top: 8px; }
.att-cell { border-radius: 8px; padding: 6px 3px; text-align: center;
            font-size: .68rem; font-weight: 600; line-height: 1.4; }
.att-na  { background: #F1F5F9; color: #94A3B8; }
.att-low { background: #FEE2E2; color: #991B1B; }
.att-mid { background: #FEF9C3; color: #713F12; }
.att-ok  { background: #DCFCE7; color: #14532D; }

/* ── Section headings ─────────────────────────────────────────────────────── */
.section-head {
    font-size: .98rem; font-weight: 700; color: #1E3A8A;
    border-left: 4px solid #2563EB; padding-left: 10px; margin: 22px 0 12px;
}

/* ── Upload hint box ──────────────────────────────────────────────────────── */
.upload-hint {
    background: #EFF6FF; border: 2px dashed #93C5FD;
    border-radius: 12px; padding: 22px; text-align: center;
    color: #1D4ED8; font-size: .9rem; margin-bottom: 16px;
}

/* ── Advice boxes ─────────────────────────────────────────────────────────── */
.advice-pass { background:#D1FAE5; border-radius:10px; padding:12px 16px;
               color:#065F46; font-size:.87rem; font-weight:600; }
.advice-warn { background:#FEF3C7; border-radius:10px; padding:12px 16px;
               color:#92400E; font-size:.87rem; font-weight:600; }
.advice-drop { background:#FEE2E2; border-radius:10px; padding:12px 16px;
               color:#991B1B; font-size:.87rem; font-weight:600; }

/* ── Input form card ──────────────────────────────────────────────────────── */
.form-section { background:#F8FAFC; border-radius:12px; padding:18px 20px;
                border:1px solid #E2E8F0; margin-bottom:12px; }
.form-section-title { font-size:.8rem; font-weight:700; text-transform:uppercase;
                      letter-spacing:1.2px; color:#475569; margin-bottom:12px; }

/* ── Tab overrides ────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button { font-weight: 600 !important; font-size: .88rem !important; }

/* ── Error banner ─────────────────────────────────────────────────────────── */
.error-banner { background:#FEF2F2; border:1.5px solid #FECACA; border-radius:12px;
                padding:18px 22px; color:#991B1B; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model_bundle.pkl"

ATT_COLS = [f"Att_Subject_{i:02d}" for i in range(1, 41)]

STATUS_VI    = {0: "Đạt",     1: "Cảnh báo",        2: "Buộc thôi học"}
STATUS_EN    = {0: "Pass",    1: "Academic Warning", 2: "Dropout Risk"}
STATUS_ICON  = {0: "✅",      1: "⚠️",               2: "❌"}
STATUS_COLOR = {0: "#10B981", 1: "#F59E0B",          2: "#EF4444"}
CLASS_COLORS = ["#10B981", "#F59E0B", "#EF4444"]

IMPORTANT_FEATURES = ["Training_Score_Mixed", "Count_F", "Tuition_Debt", "Age"]
TEXT_COLS = ["Advisor_Notes", "Personal_Essay"]
CAT_COLS  = ["Gender", "Admission_Mode", "English_Level", "Club_Member"]

ENGLISH_OPTIONS = [
    "A1","A2","B1","B2","C1","C2",
    "IELTS 4.5","IELTS 5.0","IELTS 5.5","IELTS 6.0","IELTS 6.0+","IELTS 6.5","IELTS 7.0","IELTS 7.0+",
    "TOEIC 450","TOEIC 500","TOEIC 600","TOEIC 700","TOEIC 800",
]
ADMISSION_OPTIONS = ["Thi THPT","Tuyển thẳng","ĐGNL","Xét học bạ","Xét tuyển thẳng"]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang tải model...")
def load_model() -> dict | None:
    """
    Load the serialised model bundle. Returns None on any error so the
    app can display a friendly message instead of crashing.
    """
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as fh:
            bundle = pickle.load(fh)
        # Sanity-check required keys
        required = {"model", "imputer", "text_transformers", "feature_names"}
        missing  = required - bundle.keys()
        if missing:
            st.error(f"❌ Bundle is missing keys: {missing}")
            return None
        return bundle
    except Exception as exc:
        st.error(f"❌ Failed to load model_bundle.pkl: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Đang đọc file...")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes into a DataFrame."""
    return pd.read_csv(io.BytesIO(file_bytes))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (must stay in sync with train_and_save.py)
# ─────────────────────────────────────────────────────────────────────────────
def _english_rank(level: str) -> int:
    order = {
        "A1": 1, "A2": 2, "B1": 3, "B2": 4, "B2.": 4, "C1": 5, "C2": 6,
        "IELTS 4.5": 4, "IELTS 5.0": 4, "IELTS 5.5": 5,
        "IELTS 6.0": 5, "IELTS 6.0+": 6, "IELTS 6.5": 6,
        "IELTS 7.0": 7, "IELTS 7.0+": 7,
        "TOEIC 450": 3, "TOEIC 500": 3, "TOEIC 600": 4,
        "TOEIC 700": 5, "TOEIC 800": 6,
    }
    return order.get(str(level).strip(), 0)


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 29 tabular engineered features from raw DataFrame."""
    att = df[ATT_COLS].copy()
    att.replace(-1, np.nan, inplace=True)
    att[att > 20] = np.nan
    att[att < 0]  = np.nan

    f = pd.DataFrame(index=df.index)
    f["att_mean"]        = att.mean(axis=1)
    f["att_std"]         = att.std(axis=1)
    f["att_min"]         = att.min(axis=1)
    f["att_max"]         = att.max(axis=1)
    f["att_median"]      = att.median(axis=1)
    f["att_count_valid"] = att.notna().sum(axis=1)
    f["att_count_low"]   = (att < 8).sum(axis=1)
    f["att_count_high"]  = (att >= 12).sum(axis=1)
    f["att_pct_low"]     = f["att_count_low"]  / (f["att_count_valid"] + 1e-6)
    f["att_pct_high"]    = f["att_count_high"] / (f["att_count_valid"] + 1e-6)
    f["att_fail_rate"]   = (att < 5).sum(axis=1) / (f["att_count_valid"] + 1e-6)
    f["att_sum"]         = att.sum(axis=1)
    f["att_range"]       = f["att_max"] - f["att_min"]
    early = att[ATT_COLS[:10]].mean(axis=1)
    late  = att[ATT_COLS[-10:]].mean(axis=1)
    f["att_trend"]       = late - early
    f["training_score"]  = df["Training_Score_Mixed"].fillna(50.0)
    f["count_f"]         = df["Count_F"].fillna(0)
    f["tuition_debt"]    = df["Tuition_Debt"].fillna(0)
    f["has_debt"]        = (f["tuition_debt"] > 0).astype(int)
    f["age"]             = df["Age"]
    f["english_rank"]    = df["English_Level"].apply(_english_rank)
    f["club_member"]     = (df["Club_Member"].str.strip() == "Yes").astype(int)
    f["score_x_att"]     = f["training_score"] * f["att_mean"]
    f["countf_x_attlow"] = f["count_f"] * f["att_pct_low"]
    f["hometown_ha_noi"] = df["Hometown"].str.contains("Hà Nội|Ha Noi", na=False).astype(int)
    f["addr_ha_noi"]     = df["Current_Address"].str.contains("Hà Nội|Ha Noi", na=False).astype(int)
    f["same_city"]       = (f["hometown_ha_noi"] == f["addr_ha_noi"]).astype(int)
    adm_vals = sorted(["Thi THPT", "Tuyển thẳng", "ĐGNL", "Xét học bạ", "Xét tuyển thẳng"])
    adm_map  = {m: i for i, m in enumerate(adm_vals)}
    f["admission_mode"]  = df["Admission_Mode"].map(adm_map).fillna(-1)
    f["gender"]          = (df["Gender"].str.strip() == "Nam").astype(int)
    f["risk_score"]      = (
        f["count_f"] * 2 + f["has_debt"] +
        f["att_pct_low"] * 3 - f["english_rank"] * 0.5
    )
    return f


def transform_text(df: pd.DataFrame, text_transformers: dict) -> pd.DataFrame:
    """Apply fitted TF-IDF + SVD transformers to text columns."""
    parts = []
    for col, t in text_transformers.items():
        c = df[col].fillna("")
        X = sp.hstack([t["tfidf_c"].transform(c), t["tfidf_w"].transform(c)])
        X_svd    = t["svd"].transform(X)
        svd_cols = [f"{col}_svd_{i}" for i in range(X_svd.shape[1])]
        parts.append(pd.DataFrame(X_svd, columns=svd_cols, index=df.index))
        parts.append(pd.DataFrame({
            f"{col}_len":     c.str.len().values,
            f"{col}_has_neg": c.str.contains(
                "không|bỏ|nghỉ|muộn|tụt|kém", case=False, na=False
            ).astype(int).values,
            f"{col}_has_pos": c.str.contains(
                "tốt|chăm|giỏi|xuất|đúng giờ", case=False, na=False
            ).astype(int).values,
        }, index=df.index))
    return pd.concat(parts, axis=1)


def predict_from_df(df: pd.DataFrame, bundle: dict):
    """
    Full prediction pipeline:
        raw DataFrame → feature engineering → imputation → model.predict().
    Returns (preds: ndarray[int], probas: ndarray[float, shape (N,3)]).
    """
    df = df.copy()
    # Fill missing columns with safe defaults
    for col in ATT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["Advisor_Notes", "Personal_Essay", "Hometown", "Current_Address",
                "Admission_Mode", "English_Level", "Club_Member", "Gender"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["Training_Score_Mixed", "Age"]:
        if col not in df.columns:
            df[col] = 50
    for col in ["Count_F", "Tuition_Debt"]:
        if col not in df.columns:
            df[col] = 0.0

    tab  = build_tabular_features(df).reset_index(drop=True)
    txt  = transform_text(df.reset_index(drop=True), bundle["text_transformers"])
    X    = pd.concat([tab, txt], axis=1).astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Align to training feature space
    for c in bundle["feature_names"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[bundle["feature_names"]]

    X_np   = bundle["imputer"].transform(X)
    preds  = bundle["model"].predict(X_np)
    probas = bundle["model"].predict_proba(X_np)
    return preds, probas


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _metric_tile(value: str, label: str, accent: str = "#2563EB") -> str:
    return (
        f"<div class='metric-tile' style='--accent:{accent}'>"
        f"<div class='val'>{value}</div>"
        f"<div class='lbl'>{label}</div>"
        f"</div>"
    )


def _prob_bar(label: str, pct: float, color: str) -> str:
    return (
        f"<div class='prob-row'>"
        f"  <div class='prob-label'><span>{label}</span>"
        f"  <span style='color:{color}'>{pct:.1f}%</span></div>"
        f"  <div class='prob-bar-bg'>"
        f"    <div class='prob-bar-fill' style='width:{pct:.1f}%;background:{color}'></div>"
        f"  </div>"
        f"</div>"
    )


def _att_cell(subj_num: int, val) -> str:
    label = f"S{subj_num:02d}"
    if pd.isna(val) or float(val) == -1:
        return f"<div class='att-cell att-na'>{label}<br>—</div>"
    v   = float(val)
    cls = "att-low" if v < 5 else "att-mid" if v < 10 else "att-ok"
    return f"<div class='att-cell {cls}'>{label}<br>{int(v)}</div>"


def _plotly_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,Segoe UI,sans-serif", color="#334155", size=12),
        margin=dict(t=40, b=20, l=10, r=10),
        height=height,
    )
    return fig


def _render_prediction_result(pred: int, proba: np.ndarray) -> None:
    """Shared rendering block used by both Single Student and Manual Input tabs."""
    st.markdown("<div class='section-head'>🎯 Kết quả dự đoán</div>",
                unsafe_allow_html=True)

    res_left, res_right = st.columns([1, 1], gap="large")
    class_labels = ["Đạt (Pass)", "Cảnh báo (Warning)", "Buộc thôi học (Dropout)"]

    # Left: tiles + advice
    with res_left:
        conf_pct = float(proba[pred]) * 100
        risk_pct = float(proba[1] + proba[2]) * 100
        badge    = f"<span class='badge badge-{pred}'>{STATUS_ICON[pred]} {STATUS_VI[pred]}</span>"
        tiles_html = (
            "<div class='metric-grid'>"
            + _metric_tile(badge, "Tình trạng học tập", STATUS_COLOR[pred])
            + _metric_tile(f"{conf_pct:.1f}%", "Độ tin cậy", "#2563EB")
            + _metric_tile(f"{risk_pct:.1f}%", "Xác suất rủi ro", "#EF4444")
            + "</div>"
        )
        advice_map = {
            0: "<div class='advice-pass'>🌟 Sinh viên đang học tốt. Tiếp tục phát huy!</div>",
            1: "<div class='advice-warn'>⚠️ Cảnh báo học tập. Cần tăng điểm danh, giảm môn F và chú ý nợ học phí.</div>",
            2: "<div class='advice-drop'>🚨 Nguy cơ buộc thôi học cao. Cần can thiệp khẩn cấp từ cố vấn học tập.</div>",
        }
        st.markdown(
            f"<div class='card'><div class='card-title'>Kết luận</div>"
            f"{tiles_html}<div style='height:10px'></div>{advice_map[pred]}</div>",
            unsafe_allow_html=True,
        )

    # Right: probability bars + bar chart
    with res_right:
        bars_html = "".join(
            _prob_bar(lbl, float(p) * 100, c)
            for lbl, p, c in zip(class_labels, proba, CLASS_COLORS)
        )
        st.markdown(
            f"<div class='card'><div class='card-title'>Xác suất từng lớp</div>"
            f"{bars_html}</div>",
            unsafe_allow_html=True,
        )
        fig = go.Figure(go.Bar(
            x=class_labels,
            y=[float(p) * 100 for p in proba],
            marker_color=CLASS_COLORS,
            text=[f"{float(p)*100:.1f}%" for p in proba],
            textposition="outside",
            marker_line_width=0,
        ))
        fig = _plotly_theme(fig, height=290)
        fig.update_layout(
            yaxis=dict(title="Xác suất (%)", range=[0, 118],
                       gridcolor="#E2E8F0", zeroline=False),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confidence gauge (full-width)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf_pct,
        number={"suffix": "%", "font": {"size": 28, "color": STATUS_COLOR[pred]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": STATUS_COLOR[pred], "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "#FEE2E2"},
                {"range": [40, 70],  "color": "#FEF9C3"},
                {"range": [70, 100], "color": "#D1FAE5"},
            ],
            "threshold": {
                "line": {"color": "#1E3A8A", "width": 3},
                "thickness": 0.75,
                "value": conf_pct,
            },
        },
        title={"text": "Confidence Score", "font": {"size": 14, "color": "#64748B"}},
    ))
    fig_gauge.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10, l=30, r=30),
    )
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  —  model info + quick metrics
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(bundle: dict | None) -> None:
    with st.sidebar:
        st.markdown("## 🎓 Model Info")
        st.markdown("---")

        if bundle is None:
            st.warning("⚠️ Model not loaded")
            st.markdown(
                "**model_bundle.pkl** not found.\n\n"
                "Run `train_and_save.py` first:\n"
                "```\npython train_and_save.py --data train.csv\n```"
            )
            return

        cv_f1      = bundle.get("cv_f1", 0.0)
        n_features = bundle.get("n_features", len(bundle.get("feature_names", [])))
        n_train    = bundle.get("n_train", "—")
        cv_scores  = bundle.get("cv_scores", [])

        st.markdown(
            f"<div class='sidebar-metric'>"
            f"📊 <strong>CV Macro F1</strong><br>"
            f"<strong style='font-size:1.4rem'>{cv_f1:.4f}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='sidebar-metric'>"
            f"🧠 <strong>Model</strong><br>HistGradientBoosting Ensemble (×3)"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='sidebar-metric'>"
            f"📐 <strong>Features</strong><br>{n_features} engineered features"
            f"</div>",
            unsafe_allow_html=True,
        )
        if n_train != "—":
            st.markdown(
                f"<div class='sidebar-metric'>"
                f"👥 <strong>Training samples</strong><br>{n_train:,}"
                f"</div>",
                unsafe_allow_html=True,
            )

        if cv_scores:
            st.markdown("---")
            st.markdown("**CV Fold Scores**")
            fig_cv = go.Figure(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                y=cv_scores,
                marker_color="#38BDF8",
                text=[f"{s:.3f}" for s in cv_scores],
                textposition="outside",
            ))
            fig_cv.update_layout(
                height=180,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e7ff", size=10),
                margin=dict(t=30, b=10, l=10, r=10),
                yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,.15)"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                showlegend=False,
            )
            st.plotly_chart(fig_cv, use_container_width=True)

        st.markdown("---")
        st.markdown("**Classes**")
        for k, v in STATUS_VI.items():
            icon = STATUS_ICON[k]
            st.markdown(f"{icon} **{k}** — {v}")

        st.markdown("---")
        st.caption("Academic Status Predictor · v2.0")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE STUDENT PREDICTION  (from CSV)
# ─────────────────────────────────────────────────────────────────────────────
def show_single_student(df: pd.DataFrame, bundle: dict) -> None:
    st.markdown("<div class='section-head'>🎯 Chọn sinh viên</div>",
                unsafe_allow_html=True)

    if "Student_ID" not in df.columns:
        st.error("Không tìm thấy cột `Student_ID` trong file CSV.")
        return

    ids = df["Student_ID"].astype(str).tolist()

    sel_col, btn_col = st.columns([3, 1], gap="medium")
    with sel_col:
        chosen_id = st.selectbox(
            "Chọn Student ID", options=ids,
            label_visibility="collapsed",
            help="Chọn một sinh viên để xem thông tin và dự đoán tình trạng",
        )
    with btn_col:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        do_predict = st.button(
            "🔮 Predict This Student", type="primary",
            use_container_width=True, key="single_predict",
        )

    row = df[df["Student_ID"].astype(str) == chosen_id].iloc[0]

    # ── Profile cards ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>📋 Hồ sơ sinh viên</div>",
                unsafe_allow_html=True)
    col_basic, col_acad = st.columns(2, gap="large")

    with col_basic:
        fields = [
            ("🪪 Student ID",        row.get("Student_ID", "—")),
            ("👤 Giới tính",         row.get("Gender", "—")),
            ("🎂 Tuổi",              row.get("Age", "—")),
            ("🏠 Quê quán",          row.get("Hometown", "—")),
            ("📍 Địa chỉ hiện tại",  str(row.get("Current_Address", "—"))[:55]),
            ("🎓 Tuyển sinh",        row.get("Admission_Mode", "—")),
            ("🌐 Tiếng Anh",         row.get("English_Level", "—")),
            ("🏛️ CLB",               row.get("Club_Member", "—")),
        ]
        rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fields)
        st.markdown(
            f"<div class='card'><div class='card-title'>Thông tin cơ bản</div>"
            f"<table class='info-table'><tbody>{rows_html}</tbody></table></div>",
            unsafe_allow_html=True,
        )

    with col_acad:
        att_vals = [
            float(row.get(c, np.nan))
            for c in ATT_COLS
            if pd.notna(row.get(c, np.nan))
            and float(row.get(c, np.nan)) not in (-1, 99, 100)
            and float(row.get(c, np.nan)) >= 0
        ]
        att_mean_s = f"{np.mean(att_vals):.1f}" if att_vals else "—"
        att_low_n  = sum(1 for v in att_vals if v < 8)
        debt_raw   = row.get("Tuition_Debt", 0)
        debt_s     = (
            f"{int(float(debt_raw)):,} VNĐ"
            if pd.notna(debt_raw) and float(debt_raw) > 0
            else "Không có"
        )
        acad_fields = [
            ("📊 Training Score",    row.get("Training_Score_Mixed", "—")),
            ("❌ Số môn F (hỏng)",   row.get("Count_F", "—")),
            ("💸 Nợ học phí",        debt_s),
            ("📚 Số môn đã học",     len(att_vals)),
            ("📉 ĐTB điểm danh",     att_mean_s),
            ("⚠️ Môn điểm danh thấp", att_low_n),
        ]
        rows_html2 = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in acad_fields)
        st.markdown(
            f"<div class='card'><div class='card-title'>Chỉ số học tập</div>"
            f"<table class='info-table'><tbody>{rows_html2}</tbody></table></div>",
            unsafe_allow_html=True,
        )

    # ── Attendance heatmap ─────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>📅 Bản đồ điểm danh từng môn</div>",
                unsafe_allow_html=True)
    cells = "".join(
        _att_cell(i + 1, row.get(c, np.nan)) for i, c in enumerate(ATT_COLS)
    )
    st.markdown(
        f"<div class='card'><div class='att-grid'>{cells}</div>"
        f"<div style='margin-top:10px;font-size:.73rem;color:#94A3B8'>"
        f"🟥 &lt;5 &nbsp; 🟨 5–9 &nbsp; 🟩 ≥10 &nbsp; ⬜ Chưa học</div></div>",
        unsafe_allow_html=True,
    )

    # ── Text notes ─────────────────────────────────────────────────────────────
    txt_left, txt_right = st.columns(2, gap="large")
    with txt_left:
        note = str(row.get("Advisor_Notes", "") or "*(Không có)*")
        st.markdown(
            f"<div class='card'><div class='card-title'>📝 Nhận xét cố vấn học tập</div>"
            f"<p style='font-size:.86rem;color:#334155;line-height:1.75'>{note}</p></div>",
            unsafe_allow_html=True,
        )
    with txt_right:
        essay = str(row.get("Personal_Essay", "") or "*(Không có)*")
        st.markdown(
            f"<div class='card'><div class='card-title'>✍️ Bài luận cá nhân</div>"
            f"<p style='font-size:.86rem;color:#334155;line-height:1.75'>{essay}</p></div>",
            unsafe_allow_html=True,
        )

    # ── Prediction result ──────────────────────────────────────────────────────
    if not do_predict:
        return

    with st.spinner("Đang phân tích dữ liệu và dự đoán…"):
        student_df    = df[df["Student_ID"].astype(str) == chosen_id].copy()
        preds, probas = predict_from_df(student_df, bundle)
        pred          = int(preds[0])
        proba         = probas[0]

    _render_prediction_result(pred, proba)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MANUAL INPUT PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def show_manual_input(bundle: dict) -> None:
    """
    Allows users to manually enter feature values via UI widgets and run
    a real-time prediction — no CSV file needed.
    """
    st.markdown("<div class='section-head'>✏️ Nhập thông tin sinh viên thủ công</div>",
                unsafe_allow_html=True)
    st.info(
        "💡 Điền các thông tin bên dưới. Các trường điểm danh có thể để trống "
        "(mặc định: không học môn đó).",
        icon="ℹ️",
    )

    # ── Section 1: Demographics ────────────────────────────────────────────────
    st.markdown("<div class='form-section'>"
                "<div class='form-section-title'>👤 Thông tin cá nhân</div>",
                unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        gender = st.selectbox("Giới tính", ["Nam", "Nữ"], key="m_gender")
    with d2:
        age = st.number_input("Tuổi", min_value=16, max_value=40, value=20, key="m_age")
    with d3:
        english_level = st.selectbox("Trình độ Tiếng Anh", ENGLISH_OPTIONS,
                                     index=2, key="m_english")
    with d4:
        club_member = st.selectbox("Thành viên CLB", ["No", "Yes"], key="m_club")

    d5, d6 = st.columns(2)
    with d5:
        admission_mode = st.selectbox("Phương thức tuyển sinh",
                                      ADMISSION_OPTIONS, key="m_admission")
    with d6:
        hometown = st.text_input("Quê quán (gõ 'Hà Nội' nếu ở HN)",
                                  value="Hà Nội", key="m_hometown")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: Academic metrics ────────────────────────────────────────────
    st.markdown("<div class='form-section'>"
                "<div class='form-section-title'>📊 Chỉ số học tập</div>",
                unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        training_score = st.number_input("Training Score", 0.0, 100.0, 65.0,
                                          step=0.5, key="m_score")
    with a2:
        count_f = st.number_input("Số môn F (hỏng)", 0, 30, 0, key="m_countf")
    with a3:
        tuition_debt = st.number_input("Nợ học phí (VNĐ)", 0, 50_000_000,
                                        0, step=500_000, key="m_debt")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Attendance per subject ─────────────────────────────────────
    st.markdown("<div class='form-section'>"
                "<div class='form-section-title'>📅 Điểm danh từng môn (để trống = không học)</div>",
                unsafe_allow_html=True)
    att_values = {}
    n_cols_att = 8
    rows_att   = [ATT_COLS[i:i+n_cols_att] for i in range(0, len(ATT_COLS), n_cols_att)]
    for row_cols in rows_att:
        cols = st.columns(len(row_cols))
        for ci, col_name in enumerate(row_cols):
            subj_num = int(col_name.split("_")[-1])
            with cols[ci]:
                val = st.number_input(
                    f"S{subj_num:02d}", min_value=-1, max_value=20,
                    value=-1, step=1, key=f"att_{subj_num}",
                    help="-1 = không học môn này",
                )
            att_values[col_name] = val if val != -1 else np.nan
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: Text notes ──────────────────────────────────────────────────
    st.markdown("<div class='form-section'>"
                "<div class='form-section-title'>📝 Ghi chú (tuỳ chọn)</div>",
                unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        advisor_notes = st.text_area("Nhận xét cố vấn học tập", height=100,
                                      key="m_notes")
    with t2:
        personal_essay = st.text_area("Bài luận cá nhân", height=100,
                                       key="m_essay")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Predict button ─────────────────────────────────────────────────────────
    if not st.button("🔮 Dự đoán ngay", type="primary",
                     use_container_width=True, key="manual_predict"):
        return

    # Build a single-row DataFrame matching the expected raw input schema
    row_data: dict = {
        "Student_ID":           "MANUAL_INPUT",
        "Gender":               gender,
        "Age":                  age,
        "English_Level":        english_level,
        "Club_Member":          club_member,
        "Admission_Mode":       admission_mode,
        "Hometown":             hometown,
        "Current_Address":      hometown,   # same as hometown for manual input
        "Training_Score_Mixed": training_score,
        "Count_F":              count_f,
        "Tuition_Debt":         float(tuition_debt),
        "Advisor_Notes":        advisor_notes,
        "Personal_Essay":       personal_essay,
        **att_values,
    }
    input_df = pd.DataFrame([row_data])

    with st.spinner("Đang dự đoán…"):
        try:
            preds, probas = predict_from_df(input_df, bundle)
            pred          = int(preds[0])
            proba         = probas[0]
        except Exception as exc:
            st.error(f"❌ Lỗi khi dự đoán: {exc}")
            return

    _render_prediction_result(pred, proba)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def show_batch_prediction(bundle: dict) -> None:
    st.markdown("<div class='section-head'>📁 Dự đoán hàng loạt từ file CSV</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='upload-hint'>"
        "Upload file CSV có cùng cấu trúc với <b>test.csv</b>. "
        "App sẽ predict toàn bộ và cho tải file <code>submission.csv</code>."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"],
                                 key="batch_upload", label_visibility="collapsed")
    if uploaded is None:
        return

    df_batch   = load_data(uploaded.read())
    n_students = len(df_batch)

    st.markdown(
        f"<div class='card'><div class='card-title'>"
        f"Preview file — {n_students:,} sinh viên · {df_batch.shape[1]} cột</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df_batch.head(8), use_container_width=True, height=240)
    st.markdown("</div>", unsafe_allow_html=True)

    if not st.button("🔮 Dự đoán toàn bộ", type="primary", key="batch_btn"):
        return

    all_rows   = []
    batch_size = 100
    progress   = st.progress(0, text="Đang xử lý…")

    for i in range(0, n_students, batch_size):
        chunk         = df_batch.iloc[i : i + batch_size].copy()
        preds, probas = predict_from_df(chunk, bundle)
        for j, (p, pb) in enumerate(zip(preds, probas)):
            sid = str(chunk.iloc[j].get("Student_ID", f"S{i+j}"))
            all_rows.append({
                "Student_ID":      sid,
                "Academic_Status": int(p),
                "Status_Label":    STATUS_EN[int(p)],
                "Prob_Pass":       round(float(pb[0]), 4),
                "Prob_Warning":    round(float(pb[1]), 4),
                "Prob_Dropout":    round(float(pb[2]), 4),
                "Confidence_%":    round(float(pb.max()) * 100, 1),
            })
        progress.progress(
            min((i + batch_size) / n_students, 1.0),
            text=f"Đã xử lý {min(i + batch_size, n_students):,}/{n_students:,}…",
        )

    progress.empty()
    result_df = pd.DataFrame(all_rows)

    # Summary tiles
    st.markdown("<div class='section-head'>📊 Tổng quan kết quả</div>",
                unsafe_allow_html=True)
    n_pass   = int((result_df["Academic_Status"] == 0).sum())
    n_warn   = int((result_df["Academic_Status"] == 1).sum())
    n_drop   = int((result_df["Academic_Status"] == 2).sum())
    avg_conf = result_df["Confidence_%"].mean()

    tiles_html = (
        "<div class='metric-grid'>"
        + _metric_tile(f"{n_students:,}", "Tổng sinh viên",   "#2563EB")
        + _metric_tile(f"{n_pass:,}",     "✅ Đạt",            "#10B981")
        + _metric_tile(f"{n_warn:,}",     "⚠️ Cảnh báo",       "#F59E0B")
        + _metric_tile(f"{n_drop:,}",     "❌ Buộc thôi học", "#EF4444")
        + _metric_tile(f"{avg_conf:.1f}%","Avg Confidence",   "#8B5CF6")
        + "</div>"
    )
    st.markdown(f"<div class='card'>{tiles_html}</div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        fig_pie = go.Figure(go.Pie(
            labels=["Đạt", "Cảnh báo", "Buộc thôi học"],
            values=[n_pass, n_warn, n_drop],
            marker_colors=CLASS_COLORS,
            hole=0.46, textinfo="label+percent", textfont_size=13,
        ))
        fig_pie = _plotly_theme(fig_pie, height=320)
        fig_pie.update_layout(showlegend=False,
                               title_text="Phân phối kết quả", title_x=0.5)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        fig_conf = px.histogram(result_df, x="Confidence_%", nbins=20,
                                 color_discrete_sequence=["#2563EB"],
                                 labels={"Confidence_%": "Confidence (%)"})
        fig_conf = _plotly_theme(fig_conf, height=320)
        fig_conf.update_layout(
            title_text="Phân phối Confidence", title_x=0.5,
            xaxis=dict(gridcolor="#E2E8F0", title="Confidence (%)"),
            yaxis=dict(gridcolor="#E2E8F0", title="Số sinh viên"),
            showlegend=False,
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_conf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-head'>📋 Chi tiết kết quả</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True, height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    csv_out = result_df[["Student_ID", "Academic_Status"]].to_csv(index=False)
    st.download_button(
        label="⬇️ Tải xuống submission.csv",
        data=csv_out, file_name="submission.csv",
        mime="text/csv", use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DATASET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def show_analysis(df: pd.DataFrame) -> None:
    # Overview
    st.markdown("<div class='section-head'>📈 Tổng quan dataset</div>",
                unsafe_allow_html=True)
    n_stu  = len(df)
    n_feat = df.shape[1]
    n_miss = int(df.isnull().sum().sum())
    miss_p = round(n_miss / (n_stu * n_feat) * 100, 1)
    att_ok = (df[ATT_COLS].replace(-1, np.nan)
              .apply(lambda s: s.notna() & s.between(0, 20)))
    avg_subj = att_ok.sum(axis=1).mean()

    st.markdown(
        "<div class='card'><div class='metric-grid'>"
        + _metric_tile(f"{n_stu:,}",     "Tổng sinh viên",    "#2563EB")
        + _metric_tile(f"{n_feat}",       "Số cột",            "#8B5CF6")
        + _metric_tile(f"{n_miss:,}",     "Giá trị missing",   "#F59E0B")
        + _metric_tile(f"{miss_p}%",      "Tỷ lệ missing",     "#EF4444")
        + _metric_tile(f"{avg_subj:.1f}", "Môn TB / SV",       "#10B981")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # Missing values
    st.markdown("<div class='section-head'>🕳️ Phân tích Missing Values</div>",
                unsafe_allow_html=True)
    miss_s = (df.isnull().mean() * 100).sort_values(ascending=True)
    miss_s = miss_s[miss_s > 0]
    if not miss_s.empty:
        top_miss   = miss_s.tail(20)
        bar_colors = [
            "#EF4444" if v > 50 else "#F59E0B" if v > 20 else "#3B82F6"
            for v in top_miss.values
        ]
        fig_miss = go.Figure(go.Bar(
            x=top_miss.values, y=top_miss.index.tolist(), orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in top_miss.values],
            textposition="outside",
        ))
        fig_miss = _plotly_theme(fig_miss, height=max(340, len(top_miss) * 22))
        fig_miss.update_layout(
            title_text="Top 20 cột có nhiều missing nhất", title_x=0.5,
            xaxis=dict(title="Missing (%)", gridcolor="#E2E8F0", range=[0, 115]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_miss, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Dataset không có missing values!")

    # Numeric feature distributions
    st.markdown("<div class='section-head'>📊 Phân phối features quan trọng</div>",
                unsafe_allow_html=True)
    num_present = [c for c in IMPORTANT_FEATURES if c in df.columns]
    if num_present:
        n_cols_show = min(len(num_present), 2)
        dist_cols   = st.columns(n_cols_show, gap="large")
        for idx, col_name in enumerate(num_present):
            series = df[col_name].dropna()
            if series.empty:
                continue
            fig_h = px.histogram(series, nbins=30,
                                  color_discrete_sequence=["#2563EB"],
                                  labels={col_name: col_name})
            fig_h = _plotly_theme(fig_h, height=260)
            fig_h.update_layout(
                title_text=col_name, title_x=0.5,
                xaxis=dict(gridcolor="#E2E8F0"),
                yaxis=dict(gridcolor="#E2E8F0", title="Số sinh viên"),
                showlegend=False,
            )
            with dist_cols[idx % n_cols_show]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.plotly_chart(fig_h, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Attendance distribution
    st.markdown("<div class='section-head'>📅 Phân tích điểm danh</div>",
                unsafe_allow_html=True)
    att_flat = df[ATT_COLS].values.flatten()
    att_flat = att_flat[~np.isnan(att_flat)]
    att_flat = att_flat[(att_flat >= 0) & (att_flat <= 20)]

    att_l, att_r = st.columns(2, gap="large")
    with att_l:
        fig_att = px.histogram(att_flat, nbins=21,
                                color_discrete_sequence=["#10B981"],
                                labels={"value": "Điểm danh"})
        fig_att = _plotly_theme(fig_att, height=280)
        fig_att.update_layout(
            title_text="Phân phối điểm danh tất cả môn", title_x=0.5,
            xaxis=dict(gridcolor="#E2E8F0", title="Điểm danh"),
            yaxis=dict(gridcolor="#E2E8F0", title="Số lượng"),
            showlegend=False,
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_att, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with att_r:
        att_means = (df[ATT_COLS].replace(-1, np.nan)
                     .apply(lambda c: c[c.between(0, 20)].mean()))
        fig_line = go.Figure(go.Scatter(
            x=[f"S{i:02d}" for i in range(1, 41)],
            y=att_means.values,
            mode="lines+markers",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=5, color="#2563EB"),
            fill="tozeroy", fillcolor="rgba(37,99,235,.10)",
        ))
        fig_line = _plotly_theme(fig_line, height=280)
        fig_line.update_layout(
            title_text="Điểm danh TB theo từng môn", title_x=0.5,
            xaxis=dict(gridcolor="#E2E8F0", tickangle=45, title="Môn học"),
            yaxis=dict(gridcolor="#E2E8F0", title="Điểm TB"),
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Categorical distributions
    st.markdown("<div class='section-head'>🗂️ Phân phối biến phân loại</div>",
                unsafe_allow_html=True)
    cat_present = [c for c in CAT_COLS if c in df.columns]
    for i in range(0, len(cat_present), 2):
        pair     = cat_present[i : i + 2]
        cat_cols = st.columns(len(pair), gap="large")
        for ci, cat_col in enumerate(pair):
            vc = df[cat_col].value_counts().head(12)
            fig_bar = go.Figure(go.Bar(
                x=vc.index.tolist(), y=vc.values.tolist(),
                marker_color="#6366F1",
                text=vc.values.tolist(), textposition="outside",
                marker_line_width=0,
            ))
            fig_bar = _plotly_theme(fig_bar, height=270)
            fig_bar.update_layout(
                title_text=cat_col, title_x=0.5,
                xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=25),
                yaxis=dict(gridcolor="#E2E8F0", title="Số sinh viên"),
                showlegend=False,
            )
            with cat_cols[ci]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # TF-IDF top terms
    st.markdown("<div class='section-head'>💬 Top từ khoá TF-IDF từ text</div>",
                unsafe_allow_html=True)
    from sklearn.feature_extraction.text import TfidfVectorizer as _TFIDF
    text_present = [c for c in TEXT_COLS if c in df.columns]
    if text_present:
        tf_cols = st.columns(len(text_present), gap="large")
        for ci, tc in enumerate(text_present):
            try:
                corpus = df[tc].fillna("").tolist()
                tv     = _TFIDF(analyzer="word", ngram_range=(1, 2),
                                 max_features=500, sublinear_tf=True, min_df=3)
                tv.fit(corpus)
                Xmat   = tv.transform(corpus)
                scores = np.asarray(Xmat.sum(axis=0)).flatten()
                vocab  = tv.get_feature_names_out()
                top20  = scores.argsort()[::-1][:20]
                words  = [vocab[k] for k in top20]
                vals   = [float(scores[k]) for k in top20]
                fig_tf = go.Figure(go.Bar(
                    x=vals[::-1], y=words[::-1], orientation="h",
                    marker_color="#8B5CF6",
                    text=[f"{v:.1f}" for v in vals[::-1]],
                    textposition="outside", marker_line_width=0,
                ))
                fig_tf = _plotly_theme(fig_tf, height=430)
                fig_tf.update_layout(
                    title_text=f"Top từ TF-IDF — {tc}", title_x=0.5,
                    xaxis=dict(title="TF-IDF tổng", gridcolor="#E2E8F0",
                               range=[0, max(vals) * 1.25]),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                with tf_cols[ci]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.plotly_chart(fig_tf, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as exc:
                with tf_cols[ci]:
                    st.info(f"Không thể phân tích TF-IDF cho `{tc}`: {exc}")
    else:
        st.info("Không tìm thấy cột text trong file.")

    # Correlation heatmap
    st.markdown("<div class='section-head'>🔥 Ma trận tương quan</div>",
                unsafe_allow_html=True)
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.replace(-1, np.nan)
    num_df = num_df.dropna(axis=1, thresh=int(len(df) * 0.3))
    num_df = num_df.drop(columns=ATT_COLS, errors="ignore")
    num_df = num_df.iloc[:, :12]
    if num_df.shape[1] >= 3:
        corr     = num_df.corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
        fig_corr = _plotly_theme(fig_corr, height=440)
        fig_corr.update_layout(
            title_text="Correlation Matrix (top 12 numeric features)", title_x=0.5
        )
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Không đủ cột số để vẽ correlation matrix.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Load model (may be None if file missing)
    bundle = load_model()
    cv_f1  = bundle.get("cv_f1", 0.0) if bundle else 0.0

    # Sidebar always renders
    render_sidebar(bundle)

    # Top banner
    status_note = (
        f"CV Macro F1 ≈ <b>{cv_f1:.4f}</b>" if bundle
        else "⚠️ <b>model_bundle.pkl không tìm thấy</b> — chạy train_and_save.py trước"
    )
    st.markdown(
        "<div class='top-banner'>"
        "  <div style='font-size:2.8rem;line-height:1'>🎓</div>"
        "  <div>"
        "    <h1>Academic Status Predictor</h1>"
        f"   <p>Dự đoán tình trạng học tập sinh viên &nbsp;·&nbsp; "
        f"   {status_note} &nbsp;·&nbsp; "
        "    3 lớp: <b>Đạt</b> · <b>Cảnh báo</b> · <b>Buộc thôi học</b></p>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # If model is not loaded — show clear instructions and stop
    if bundle is None:
        st.markdown(
            "<div class='error-banner'>"
            "<b>❌ model_bundle.pkl not found</b><br><br>"
            "The app cannot run without a trained model bundle. Please:<br>"
            "<ol style='margin:8px 0 0 18px'>"
            "<li>Make sure <code>train_and_save.py</code> is in the same folder.</li>"
            "<li>Run: <code>python train_and_save.py --data train.csv</code></li>"
            "<li>Ensure <code>model_bundle.pkl</code> is in the same directory as <code>app.py</code>.</li>"
            "<li>Restart the Streamlit app.</li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Shared CSV upload (Tab 1 + Tab 4)
    st.markdown(
        "<div class='section-head'>📂 Upload file test.csv (dùng cho Tab 1 & Tab 4)</div>",
        unsafe_allow_html=True,
    )
    up_col, guide_col = st.columns([2, 1], gap="large")
    with up_col:
        main_file = st.file_uploader(
            "Upload test.csv", type=["csv"],
            key="main_upload", label_visibility="collapsed",
        )
    with guide_col:
        st.markdown(
            "<div class='card' style='margin-top:0'>"
            "<div class='card-title'>Hướng dẫn sử dụng</div>"
            "<ul style='font-size:.83rem;color:#475569;padding-left:16px;margin:0'>"
            "<li><b>Tab 1</b> — Chọn Student ID → xem hồ sơ → predict</li>"
            "<li><b>Tab 2</b> — Nhập tay thông tin → predict ngay</li>"
            "<li><b>Tab 3</b> — Upload CSV riêng → batch predict → download</li>"
            "<li><b>Tab 4</b> — EDA & phân tích toàn bộ dataset</li>"
            "</ul>"
            "</div>",
            unsafe_allow_html=True,
        )

    df_main = None
    if main_file is not None:
        df_main = load_data(main_file.read())
        st.success(
            f"✅ Đã tải **{len(df_main):,}** sinh viên · {df_main.shape[1]} cột",
            icon="📊",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯  Single Student",
        "✏️  Manual Input",
        "📦  Batch Prediction",
        "📈  Dataset Analysis",
    ])

    with tab1:
        if df_main is None:
            st.markdown(
                "<div class='upload-hint'>"
                "⬆️ Vui lòng upload file <b>test.csv</b> ở trên để bắt đầu."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            show_single_student(df_main, bundle)

    with tab2:
        show_manual_input(bundle)

    with tab3:
        show_batch_prediction(bundle)

    with tab4:
        if df_main is None:
            st.markdown(
                "<div class='upload-hint'>"
                "⬆️ Vui lòng upload file <b>test.csv</b> ở trên để xem phân tích."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            show_analysis(df_main)

    # Footer
    st.markdown(
        f"<div style='text-align:center;color:#94A3B8;font-size:.76rem;padding:18px 0 8px'>"
        f"Academic Status Predictor &nbsp;·&nbsp; HistGradientBoosting Ensemble"
        f"&nbsp;·&nbsp; CV Macro F1 ≈ {cv_f1:.4f}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()