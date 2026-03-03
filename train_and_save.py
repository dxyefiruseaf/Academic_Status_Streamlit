"""
train_and_save.py
-----------------
Run this script locally (with train.csv + test.csv) to regenerate model_bundle.pkl.
Then commit the new pkl to your GitHub repo and redeploy on Streamlit Cloud.

Usage:
    python train_and_save.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import scipy.sparse as sp

RANDOM_STATE = 42
ATT_COLS = [f"Att_Subject_{i:02d}" for i in range(1, 41)]


def _english_rank(level):
    order = {
        "A1":1,"A2":2,"B1":3,"B2":4,"B2.":4,"C1":5,"C2":6,
        "IELTS 4.5":4,"IELTS 5.0":4,"IELTS 5.5":5,"IELTS 6.0":5,
        "IELTS 6.0+":6,"IELTS 6.5":6,"IELTS 7.0":7,"IELTS 7.0+":7,
        "TOEIC 450":3,"TOEIC 500":3,"TOEIC 600":4,"TOEIC 700":5,"TOEIC 800":6,
    }
    return order.get(str(level).strip(), 0)


def build_tabular_features(df):
    att = df[ATT_COLS].copy()
    att.replace(-1, np.nan, inplace=True)
    att[att > 20] = np.nan
    att[att < 0] = np.nan

    feats = pd.DataFrame(index=df.index)
    feats["att_mean"]         = att.mean(axis=1)
    feats["att_std"]          = att.std(axis=1)
    feats["att_min"]          = att.min(axis=1)
    feats["att_max"]          = att.max(axis=1)
    feats["att_median"]       = att.median(axis=1)
    feats["att_count_valid"]  = att.notna().sum(axis=1)
    feats["att_count_low"]    = (att < 8).sum(axis=1)
    feats["att_count_high"]   = (att >= 12).sum(axis=1)
    feats["att_pct_low"]      = feats["att_count_low"] / (feats["att_count_valid"] + 1e-6)
    feats["att_pct_high"]     = feats["att_count_high"] / (feats["att_count_valid"] + 1e-6)
    feats["att_fail_rate"]    = (att < 5).sum(axis=1) / (feats["att_count_valid"] + 1e-6)
    feats["att_sum"]          = att.sum(axis=1)
    feats["att_range"]        = feats["att_max"] - feats["att_min"]
    early = att[ATT_COLS[:10]].mean(axis=1)
    late  = att[ATT_COLS[-10:]].mean(axis=1)
    feats["att_trend"]        = late - early
    feats["training_score"]   = df["Training_Score_Mixed"].fillna(50.0)
    feats["count_f"]          = df["Count_F"].fillna(0)
    feats["tuition_debt"]     = df["Tuition_Debt"].fillna(0)
    feats["has_debt"]         = (feats["tuition_debt"] > 0).astype(int)
    feats["age"]              = df["Age"]
    feats["english_rank"]     = df["English_Level"].apply(_english_rank)
    feats["club_member"]      = (df["Club_Member"].str.strip() == "Yes").astype(int)
    feats["score_x_att"]      = feats["training_score"] * feats["att_mean"]
    feats["countf_x_attlow"]  = feats["count_f"] * feats["att_pct_low"]
    feats["hometown_ha_noi"]  = df["Hometown"].str.contains("Hà Nội|Ha Noi", na=False).astype(int)
    feats["addr_ha_noi"]      = df["Current_Address"].str.contains("Hà Nội|Ha Noi", na=False).astype(int)
    feats["same_city"]        = (feats["hometown_ha_noi"] == feats["addr_ha_noi"]).astype(int)
    adm_vals = sorted(["Thi THPT","Tuyển thẳng","ĐGNL","Xét học bạ","Xét tuyển thẳng"])
    adm_map  = {m: i for i, m in enumerate(adm_vals)}
    feats["admission_mode"]   = df["Admission_Mode"].map(adm_map).fillna(-1)
    feats["gender"]           = (df["Gender"].str.strip() == "Nam").astype(int)
    feats["risk_score"]       = feats["count_f"]*2 + feats["has_debt"] + feats["att_pct_low"]*3 - feats["english_rank"]*0.5
    return feats


def fit_text_transformers(train_df, test_df, n_components=15):
    transformers = {}
    for col in ["Advisor_Notes", "Personal_Essay"]:
        corpus = pd.concat([train_df[col].fillna(""), test_df[col].fillna("")], ignore_index=True)
        tfidf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=2000, sublinear_tf=True, min_df=5)
        tfidf_w = TfidfVectorizer(analyzer="word",    ngram_range=(1,2), max_features=1500, sublinear_tf=True, min_df=5)
        tfidf_c.fit(corpus)
        tfidf_w.fit(corpus)
        X_all = sp.hstack([tfidf_c.transform(corpus), tfidf_w.transform(corpus)])
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        svd.fit(X_all)
        transformers[col] = {"tfidf_c": tfidf_c, "tfidf_w": tfidf_w, "svd": svd}
    return transformers


def transform_text(df, transformers):
    parts = []
    for col, t in transformers.items():
        c = df[col].fillna("")
        X = sp.hstack([t["tfidf_c"].transform(c), t["tfidf_w"].transform(c)])
        X_svd = t["svd"].transform(X)
        cols  = [f"{col}_svd_{i}" for i in range(X_svd.shape[1])]
        parts.append(pd.DataFrame(X_svd, columns=cols, index=df.index))
        parts.append(pd.DataFrame({
            f"{col}_len":     c.str.len().values,
            f"{col}_has_neg": c.str.contains("không|bỏ|nghỉ|muộn|tụt|kém", case=False, na=False).astype(int).values,
            f"{col}_has_pos": c.str.contains("tốt|chăm|giỏi|xuất|đúng giờ", case=False, na=False).astype(int).values,
        }, index=df.index))
    return pd.concat(parts, axis=1)


def build_all_features(df, text_transformers):
    tab = build_tabular_features(df).reset_index(drop=True)
    txt = transform_text(df.reset_index(drop=True), text_transformers)
    X   = pd.concat([tab, txt], axis=1).astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X


def main():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")
    y_train  = train_df["Academic_Status"].values

    print("Fitting text transformers...")
    text_transformers = fit_text_transformers(train_df, test_df)

    print("Building features...")
    X_train = build_all_features(train_df, text_transformers)
    print(f"  Feature shape: {X_train.shape}")

    imputer = SimpleImputer(strategy="median")
    X_np    = imputer.fit_transform(X_train)

    # Cross-validation
    print("Running 5-fold CV...")
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1s  = []
    for fold, (tr, va) in enumerate(skf.split(X_np, y_train)):
        m = HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, min_samples_leaf=25,
            l2_regularization=3.0, learning_rate=0.06,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        m.fit(X_np[tr], y_train[tr])
        f1 = f1_score(y_train[va], m.predict(X_np[va]), average="macro")
        f1s.append(f1)
        print(f"  Fold {fold+1}: {f1:.4f}")
    cv_f1 = float(np.mean(f1s))
    print(f"  Mean CV F1: {cv_f1:.4f}")

    # Final model on full data
    print("Training final model...")
    final_model = HistGradientBoostingClassifier(
        max_iter=400, max_depth=5, min_samples_leaf=25,
        l2_regularization=3.0, learning_rate=0.06,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    final_model.fit(X_np, y_train)

    bundle = {
        "imputer":           imputer,
        "text_transformers": text_transformers,
        "model":             final_model,
        "feature_names":     list(X_train.columns),
        "cv_f1":             cv_f1,
        "n_classes":         3,
    }

    with open("model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    size_mb = os.path.getsize("model_bundle.pkl") / 1024 / 1024
    print(f"Saved model_bundle.pkl  ({size_mb:.1f} MB)")
    print("Done! Now commit and push to GitHub to redeploy.")


if __name__ == "__main__":
    main()
