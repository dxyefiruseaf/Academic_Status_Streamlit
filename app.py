"""
Academic Status Predictor — Streamlit Demo
Model: HistGradientBoosting + ExtraTrees Stack | CV Macro F1 ≈ 0.86
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Status Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #f4f6fb; }

    .stApp { 
        background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
    }

    .metric-card {
        background: white;
        border: 1px solid #e3e8ff;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(100, 108, 255, 0.08);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-badge-0 {
        background: linear-gradient(135deg, #22c55e, #06b6d4);
        color: white;
        border-radius: 20px;
        padding: 6px 18px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(34,197,94,0.2);
    }

    .status-badge-1 {
        background: linear-gradient(135deg, #facc15, #fb923c);
        color: white;
        border-radius: 20px;
        padding: 6px 18px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(250,204,21,0.2);
    }

    .status-badge-2 {
        background: linear-gradient(135deg, #ef4444, #f43f5e);
        color: white;
        border-radius: 20px;
        padding: 6px 18px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(239,68,68,0.2);
    }

    .predict-btn > button {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.7rem 2rem !important;
        width: 100% !important;
        box-shadow: 0 6px 18px rgba(79,70,229,0.25);
        transition: all 0.3s ease;
    }

    .predict-btn > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(79,70,229,0.35);
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        border-left: 4px solid #4f46e5;
        padding-left: 10px;
        margin: 16px 0 10px 0;
    }

    .info-box {
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #06b6d4;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ATT_COLS = [f"Att_Subject_{i:02d}" for i in range(1, 41)]
ENGLISH_OPTIONS = ["A1","A2","B1","B2","C1","C2",
                   "IELTS 5.0","IELTS 5.5","IELTS 6.0","IELTS 6.0+","IELTS 6.5","IELTS 7.0",
                   "TOEIC 450","TOEIC 500","TOEIC 600","TOEIC 700","TOEIC 800"]
ADMISSION_OPTIONS = ["Thi THPT","Tuyển thẳng","ĐGNL","Xét học bạ","Xét tuyển thẳng"]
STATUS_LABELS = {0: "✅ Đạt (Pass)", 1: "⚠️ Cảnh báo (Warning)", 2: "❌ Buộc thôi học (Dropout)"}
STATUS_COLORS = {0: "#00b894", 1: "#fdcb6e", 2: "#d63031"}
STATUS_EN     = {0: "Pass", 1: "Academic Warning", 2: "Dropout Risk"}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = Path(__file__).parent / "model_bundle.pkl"
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model()

# ── Feature helpers ───────────────────────────────────────────────────────────
def _english_rank(level):
    order = {'A1':1,'A2':2,'B1':3,'B2':4,'B2.':4,'C1':5,'C2':6,
             'IELTS 4.5':4,'IELTS 5.0':4,'IELTS 5.5':5,'IELTS 6.0':5,
             'IELTS 6.0+':6,'IELTS 6.5':6,'IELTS 7.0':7,'IELTS 7.0+':7,
             'TOEIC 450':3,'TOEIC 500':3,'TOEIC 600':4,'TOEIC 700':5,'TOEIC 800':6}
    return order.get(str(level).strip(), 0)

def build_tabular_features(df):
    att = df[ATT_COLS].copy()
    att.replace(-1, np.nan, inplace=True)
    att[att > 20] = np.nan
    att[att < 0] = np.nan

    feats = pd.DataFrame(index=df.index)
    feats['att_mean'] = att.mean(axis=1)
    feats['att_std'] = att.std(axis=1)
    feats['att_min'] = att.min(axis=1)
    feats['att_max'] = att.max(axis=1)
    feats['att_median'] = att.median(axis=1)
    feats['att_count_valid'] = att.notna().sum(axis=1)
    feats['att_count_low'] = (att < 8).sum(axis=1)
    feats['att_count_high'] = (att >= 12).sum(axis=1)
    feats['att_pct_low'] = feats['att_count_low'] / (feats['att_count_valid'] + 1e-6)
    feats['att_pct_high'] = feats['att_count_high'] / (feats['att_count_valid'] + 1e-6)
    feats['att_fail_rate'] = (att < 5).sum(axis=1) / (feats['att_count_valid'] + 1e-6)
    feats['att_sum'] = att.sum(axis=1)
    feats['att_range'] = feats['att_max'] - feats['att_min']
    early = att[ATT_COLS[:10]].mean(axis=1)
    late  = att[ATT_COLS[-10:]].mean(axis=1)
    feats['att_trend'] = late - early
    feats['training_score'] = df['Training_Score_Mixed'].fillna(50.0)
    feats['count_f'] = df['Count_F'].fillna(0)
    feats['tuition_debt'] = df['Tuition_Debt'].fillna(0)
    feats['has_debt'] = (feats['tuition_debt'] > 0).astype(int)
    feats['age'] = df['Age']
    feats['english_rank'] = df['English_Level'].apply(_english_rank)
    feats['club_member'] = (df['Club_Member'].str.strip() == 'Yes').astype(int)
    feats['score_x_att'] = feats['training_score'] * feats['att_mean']
    feats['countf_x_attlow'] = feats['count_f'] * feats['att_pct_low']
    feats['hometown_ha_noi'] = df['Hometown'].str.contains('Hà Nội|Ha Noi', na=False).astype(int)
    feats['addr_ha_noi'] = df['Current_Address'].str.contains('Hà Nội|Ha Noi', na=False).astype(int)
    feats['same_city'] = (feats['hometown_ha_noi'] == feats['addr_ha_noi']).astype(int)
    adm_vals = sorted(['Thi THPT','Tuyển thẳng','ĐGNL','Xét học bạ','Xét tuyển thẳng'])
    adm_map = {m: i for i, m in enumerate(adm_vals)}
    feats['admission_mode'] = df['Admission_Mode'].map(adm_map).fillna(-1)
    feats['gender'] = (df['Gender'].str.strip() == 'Nam').astype(int)
    feats['risk_score'] = feats['count_f']*2 + feats['has_debt'] + feats['att_pct_low']*3 - feats['english_rank']*0.5
    return feats

def transform_text_single(df, text_transformers):
    parts = []
    for col, t in text_transformers.items():
        c = df[col].fillna('')
        X = sp.hstack([t['tfidf_c'].transform(c), t['tfidf_w'].transform(c)])
        X_svd = t['svd'].transform(X)
        cols = [f'{col}_svd_{i}' for i in range(X_svd.shape[1])]
        parts.append(pd.DataFrame(X_svd, columns=cols, index=df.index))
        parts.append(pd.DataFrame({
            f'{col}_len': c.str.len().values,
            f'{col}_has_neg': c.str.contains('không|bỏ|nghỉ|muộn|tụt|kém', case=False, na=False).astype(int).values,
            f'{col}_has_pos': c.str.contains('tốt|chăm|giỏi|xuất|đúng giờ', case=False, na=False).astype(int).values,
        }, index=df.index))
    return pd.concat(parts, axis=1)

def predict_student(row_dict):
    df = pd.DataFrame([row_dict])
    for col in ATT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    tab = build_tabular_features(df).reset_index(drop=True)
    txt = transform_text_single(df.reset_index(drop=True), bundle['text_transformers'])
    X = pd.concat([tab, txt], axis=1).astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Align features
    for c in bundle['feature_names']:
        if c not in X.columns:
            X[c] = 0.0
    X = X[bundle['feature_names']]

    X_np = bundle['imputer'].transform(X)
    proba = bundle['model'].predict_proba(X_np)[0]
    pred  = int(np.argmax(proba))
    return pred, proba

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Academic Status Predictor")
    st.markdown(f"""
    <div class='info-box'>
        <b>Model:</b> HistGradientBoosting<br>
        <b>CV Macro F1:</b> {bundle['cv_f1']:.4f}<br>
        <b>Features:</b> {len(bundle['feature_names'])}<br>
        <b>Classes:</b> Pass / Warning / Dropout
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Input Mode")
    mode = st.radio("", ["🧑 Single Student", "📁 Batch CSV Upload"], label_visibility="collapsed")

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Academic Status Predictor")
st.markdown("*Dự đoán tình trạng học tập của sinh viên dựa trên điểm danh, điểm số và hồ sơ cá nhân*")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE STUDENT MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🧑 Single Student":
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='section-header'>👤 Thông tin cơ bản</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            student_id  = st.text_input("Student ID", value="SV20210001")
            gender      = st.selectbox("Giới tính", ["Nam", "Nữ"])
            age         = st.number_input("Tuổi", min_value=17, max_value=35, value=20)
        with c2:
            hometown    = st.text_input("Quê quán", value="Hà Nội")
            cur_address = st.text_input("Địa chỉ hiện tại", value="Đống Đa, Hà Nội")
            club_member = st.selectbox("Tham gia CLB", ["Yes", "No"])

        c3, c4 = st.columns(2)
        with c3:
            admission   = st.selectbox("Phương thức tuyển sinh", ADMISSION_OPTIONS)
            english     = st.selectbox("Trình độ tiếng Anh", ENGLISH_OPTIONS, index=2)
        with c4:
            training_score = st.slider("Training Score", 0, 100, 75)
            count_f     = st.number_input("Số môn F (hỏng)", min_value=0, max_value=15, value=0)
            tuition_debt = st.number_input("Nợ học phí (VNĐ)", min_value=0, step=500000, value=0)

        st.markdown("<div class='section-header'>📝 Ghi chú & Bài luận</div>", unsafe_allow_html=True)
        advisor_notes = st.text_area("Nhận xét của cố vấn học tập", height=90,
            value="Em là sinh viên chăm chỉ, luôn đúng giờ và tích cực tham gia học tập.")
        personal_essay = st.text_area("Bài luận cá nhân", height=90,
            value="Mình rất yêu thích ngành học này và luôn cố gắng học tốt mỗi ngày.")

    with col2:
        st.markdown("<div class='section-header'>📊 Điểm danh môn học (để trống = chưa học)</div>",
                    unsafe_allow_html=True)

        att_values = {}
        # Show 20 subjects in 4 columns
        cols_att = st.columns(4)
        subjects_to_show = 20
        for i in range(subjects_to_show):
            col_idx = i % 4
            subj_num = i + 1
            key = f"Att_Subject_{subj_num:02d}"
            with cols_att[col_idx]:
                val = st.number_input(
                    f"S{subj_num:02d}",
                    min_value=-1, max_value=15,
                    value=-1,
                    key=key,
                    help="-1 = chưa học môn này"
                )
                att_values[key] = float(val) if val >= 0 else np.nan

        # Remaining subjects default to NaN
        for i in range(subjects_to_show, 40):
            att_values[f"Att_Subject_{i+1:02d}"] = np.nan

        st.markdown("")
        predict_clicked = st.button("🔮 Dự đoán tình trạng", use_container_width=True, type="primary")

    # ── Prediction Result ─────────────────────────────────────────────────────
    if predict_clicked:
        row = {
            "Student_ID": student_id, "Gender": gender, "Age": age,
            "Hometown": hometown, "Current_Address": cur_address,
            "Admission_Mode": admission, "English_Level": english,
            "Club_Member": club_member, "Tuition_Debt": float(tuition_debt),
            "Count_F": float(count_f), "Training_Score_Mixed": int(training_score),
            "Advisor_Notes": advisor_notes, "Personal_Essay": personal_essay,
        }
        row.update(att_values)

        with st.spinner("Đang phân tích..."):
            pred, proba = predict_student(row)

        st.markdown("---")
        st.markdown("## 🎯 Kết quả dự đoán")

        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

        with res_col1:
            badge_html = f"<div class='status-badge-{pred}'>{STATUS_LABELS[pred]}</div>"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Tình trạng học tập</div>
                <br>{badge_html}
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            conf = float(proba[pred]) * 100
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{conf:.1f}%</div>
                <div class='metric-label'>Độ tin cậy</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            risk = float(proba[1] + proba[2]) * 100
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{risk:.1f}%</div>
                <div class='metric-label'>Xác suất rủi ro</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability chart
        st.markdown("### 📊 Phân phối xác suất")
        labels = ["Đạt (0)", "Cảnh báo (1)", "Buộc thôi học (2)"]
        colors = ["#00b894", "#fdcb6e", "#d63031"]

        fig = go.Figure(go.Bar(
            x=labels,
            y=[p * 100 for p in proba],
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in proba],
            textposition="outside",
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6", size=13),
            yaxis=dict(title="Xác suất (%)", range=[0, 110], gridcolor="#2d3561"),
            xaxis=dict(gridcolor="#2d3561"),
            showlegend=False,
            height=320,
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Advice
        st.markdown("### 💡 Khuyến nghị")
        if pred == 0:
            st.success("🌟 Sinh viên đang học tốt. Tiếp tục duy trì phong độ và phấn đấu để đạt kết quả cao hơn!")
        elif pred == 1:
            st.warning("⚠️ Sinh viên đang trong tình trạng cảnh báo học tập. Cần tăng cường điểm danh, giảm số môn F và chú ý nợ học phí.")
        else:
            st.error("🚨 Nguy cơ buộc thôi học cao. Cần can thiệp khẩn cấp: gặp cố vấn học tập, xem xét đăng ký học lại và giải quyết nợ học phí.")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 📁 Dự đoán hàng loạt từ file CSV")

    st.markdown("""
    <div class='info-box'>
        Upload file CSV có cùng cấu trúc với <b>test.csv</b>. 
        Kết quả sẽ được hiển thị và có thể tải xuống.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f"**Đã tải:** {len(df_upload)} sinh viên, {df_upload.shape[1]} cột")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("🔮 Dự đoán tất cả", type="primary"):
            with st.spinner(f"Đang dự đoán {len(df_upload)} sinh viên..."):
                results = []
                progress = st.progress(0)
                batch_size = 50
                all_rows = []

                for i in range(0, len(df_upload), batch_size):
                    chunk = df_upload.iloc[i:i+batch_size].copy()
                    for col in ATT_COLS:
                        if col not in chunk.columns:
                            chunk[col] = np.nan
                    for col in ['Advisor_Notes','Personal_Essay','Hometown','Current_Address',
                                'Admission_Mode','English_Level','Club_Member','Gender']:
                        if col not in chunk.columns:
                            chunk[col] = ''
                    for col in ['Training_Score_Mixed','Age']:
                        if col not in chunk.columns:
                            chunk[col] = 50
                    for col in ['Count_F','Tuition_Debt']:
                        if col not in chunk.columns:
                            chunk[col] = 0.0

                    tab = build_tabular_features(chunk).reset_index(drop=True)
                    txt = transform_text_single(chunk.reset_index(drop=True), bundle['text_transformers'])
                    X = pd.concat([tab, txt], axis=1).astype(np.float32)
                    X.replace([np.inf, -np.inf], np.nan, inplace=True)
                    for c in bundle['feature_names']:
                        if c not in X.columns:
                            X[c] = 0.0
                    X = X[bundle['feature_names']]
                    X_np = bundle['imputer'].transform(X)
                    preds = bundle['model'].predict(X_np)
                    probas = bundle['model'].predict_proba(X_np)

                    for j, (p, pb) in enumerate(zip(preds, probas)):
                        all_rows.append({
                            "Student_ID": chunk.iloc[j].get("Student_ID", f"S{i+j}"),
                            "Academic_Status": int(p),
                            "Status_Label": STATUS_EN[int(p)],
                            "Prob_Pass": round(float(pb[0]), 4),
                            "Prob_Warning": round(float(pb[1]), 4),
                            "Prob_Dropout": round(float(pb[2]), 4),
                            "Confidence": round(float(pb.max()) * 100, 1),
                        })
                    progress.progress(min((i + batch_size) / len(df_upload), 1.0))

            result_df = pd.DataFrame(all_rows)

            # Summary stats
            st.markdown("---")
            st.markdown("### 📊 Tổng quan kết quả")
            m1, m2, m3, m4 = st.columns(4)
            total = len(result_df)
            n_pass = (result_df["Academic_Status"] == 0).sum()
            n_warn = (result_df["Academic_Status"] == 1).sum()
            n_drop = (result_df["Academic_Status"] == 2).sum()
            avg_conf = result_df["Confidence"].mean()

            for col, val, label, color in [
                (m1, total, "Tổng sinh viên", "#6c63ff"),
                (m2, n_pass, "Đạt ✅", "#00b894"),
                (m3, n_warn, "Cảnh báo ⚠️", "#fdcb6e"),
                (m4, n_drop, "Buộc thôi học ❌", "#d63031"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color:{color}'>{val}</div>
                        <div class='metric-label'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Pie chart
            fig_pie = go.Figure(go.Pie(
                labels=["Đạt", "Cảnh báo", "Buộc thôi học"],
                values=[n_pass, n_warn, n_drop],
                marker_colors=["#00b894","#fdcb6e","#d63031"],
                hole=0.4,
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccd6f6"),
                height=300,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Full results table
            st.markdown("### 📋 Chi tiết kết quả")
            st.dataframe(result_df, use_container_width=True, height=400)

            # Download
            csv_out = result_df[["Student_ID","Academic_Status"]].to_csv(index=False)
            st.download_button(
                "⬇️ Tải xuống submission.csv",
                data=csv_out,
                file_name="submission.csv",
                mime="text/csv",
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8892b0; font-size:0.8rem'>"
    "Academic Status Predictor · HistGradientBoosting Ensemble · Macro F1 ≈ 0.86"
    "</div>",
    unsafe_allow_html=True,
)
