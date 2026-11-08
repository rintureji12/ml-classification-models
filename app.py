import os, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, confusion_matrix, classification_report, roc_curve)

STUDENT_NAME = "Developed by: RINTU REJI"
BITS_ID      = "2026AJ05184"
COURSE       = "Machine Learning – Assignment 2"

st.set_page_config(page_title="TumourScope | ML Classifier Lab", page_icon="🔬", layout="wide")
TEAL, AMBER, SAND = "#0F766E", "#D97706", "#EFE8DC"
CLASSES = ["Malignant", "Benign"]          # target: 0 = malignant, 1 = benign
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.markdown(f"""
<style>
.hero {{background: linear-gradient(120deg, {TEAL} 0%, #134E4A 60%, #1F2937 100%); padding: 28px 34px;
        border-radius: 18px; color: #fff; margin-bottom: 18px;}}
.hero h1 {{margin: 0; font-size: 2.3rem; letter-spacing: .5px; color:#fff;}}
.hero p {{margin: 4px 0 0 0; opacity: .9; font-size: 1.02rem; color:#fff;}}
.badge {{display:inline-block; background:{AMBER}; color:#fff; padding:4px 12px; border-radius:20px;
         font-size:.85rem; margin: 12px 8px 0 0;}}
.card {{background:{SAND}; border-left: 6px solid {TEAL}; border-radius: 12px; padding: 14px 16px; text-align:center;}}
.card .v {{font-size: 1.25rem; font-weight: 700; color:{TEAL};}}
.card .l {{font-size: .8rem; text-transform: uppercase; letter-spacing: 1.2px; color:#6B7280;}}
</style>
<div class="hero">
  <h1> TumourScope</h1>
  <p>Five classifiers, one question: is the tumour malignant or benign?</p>
  <span class="badge"> {STUDENT_NAME}</span><span class="badge">  BITS ID: {BITS_ID}</span>
  <span class="badge"> {COURSE}</span>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(fname):
    return joblib.load(os.path.join(BASE, "model", fname))

def metrics_for(y, pred, prob):
    try: auc = roc_auc_score(y, prob)
    except ValueError: auc = float("nan")
    return {"Accuracy": accuracy_score(y, pred), "AUC": auc,
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0), "MCC": matthews_corrcoef(y, pred)}

def card(col, label, value):
    col.markdown(f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


st.markdown("### What this app does")
st.write("This app predicts whether a breast tumour is **malignant or benign** from measurements of cell nuclei "
         "taken from a fine-needle aspirate (FNA) image. Five classification models were trained on the same "
         "dataset; upload the test file, pick a model, and see how well it classifies unseen cases.")
d1, d2, d3, d4 = st.columns(4)
card(d1, "Dataset", "Breast Cancer Wisconsin (Diagnostic)")
card(d2, "Total rows", "569")
card(d3, "Features", "30 numeric")
card(d4, "Target", "Malignant (212) / Benign (357)")
st.caption("Dataset: "
           "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic. "
           )
st.divider()

# Sidebar 
with st.sidebar:
    st.header("⚙️ Try out various models")
    uploaded = st.file_uploader("1. Upload test CSV (features + 'target')", type="csv")
    choice = st.selectbox("2. Choose a classifier", list(MODELS))
    st.download_button("⬇️ Sample test_data.csv", open(os.path.join(BASE, "test_data.csv"), "rb"),
                       "test_data.csv", mime="text/csv")

if uploaded is None:
    st.info("👈 Upload the test CSV in the sidebar to begin (use the sample file if needed).")
    st.stop()

df = pd.read_csv(uploaded)
if "target" not in df.columns:
    st.error("The CSV must contain a 'target' column (0 = malignant, 1 = benign)."); st.stop()
model = load_model(MODELS[choice])
expected = list(getattr(model, "feature_names_in_", [c for c in df.columns if c != "target"]))
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}"); st.stop()
X, y = df[expected], df["target"].astype(int)

prob = model.predict_proba(X)[:, 1]
pred = model.predict(X)
m = metrics_for(y, pred, prob)

rows = []
for name, f in MODELS.items():
    mm = load_model(f)
    rows.append({"ML Model Name": name, **metrics_for(y, mm.predict(X), mm.predict_proba(X)[:, 1])})
board = pd.DataFrame(rows).set_index("ML Model Name")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 ROC & confidence", "🏆 Leaderboard",
                                        "🔎 Case explorer"])

# Tab 1 
with tab1:
    st.subheader(f"{choice} - Evaluation metrics")
    for c, (k, v) in zip(st.columns(6), m.items()):
        card(c, k, f"{v:.4f}")
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="BuGn")
        ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        st.pyplot(fig)
    with b:
        st.markdown("**Classification report**")
        rep = classification_report(y, pred, labels=[0, 1], target_names=CLASSES, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
    st.markdown("---")
    st.subheader("Comparison of all 5 models on the uploaded test data")
    st.dataframe(board.round(4), use_container_width=True)
    st.success(f"🏆 Best by MCC: **{board['MCC'].idxmax()}** ({board['MCC'].max():.4f})")
    out = df.copy(); out["predicted"] = pred; out["P(benign)"] = prob.round(4)
    st.download_button("⬇️ Download predictions CSV", out.to_csv(index=False), "predictions.csv", "text/csv")

# Tab 2
with tab2:
    a, b = st.columns(2)
    with a:
        st.markdown("**ROC curve – all five models**")
        fig, ax = plt.subplots(figsize=(4.6, 4))
        colors = [TEAL, AMBER, "#7C3AED", "#DC2626", "#2563EB"]
        for (name, f), col in zip(MODELS.items(), colors):
            mm = load_model(f); p = mm.predict_proba(X)[:, 1]
            try:
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, color=col, lw=3 if name == choice else 1.3,
                        label=f"{name} ({roc_auc_score(y, p):.3f})")
            except ValueError:
                pass
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right"); st.pyplot(fig)
    with b:
        st.markdown(f"**Confidence distribution – {choice}**")
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ax.hist(prob[y == 0], bins=15, alpha=.75, color=AMBER, label="Actual malignant")
        ax.hist(prob[y == 1], bins=15, alpha=.75, color=TEAL, label="Actual benign")
        ax.set_xlabel("Predicted P(benign)"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
        st.pyplot(fig)

# Tab 3 
with tab3:
    st.subheader("Rank the five models by any metric")
    metric = st.radio("Rank by", list(board.columns), horizontal=True, index=5)
    st.bar_chart(board[metric].sort_values(ascending=False), color=TEAL)
    best = board[metric].idxmax()
    st.success(f"🏆 Best by {metric}: **{best}** ({board[metric].max():.4f})")

# Tab 4
with tab4:
    st.subheader("Inspect a single sample")
    idx = st.number_input("Row number", 0, len(df) - 1, 0)
    p_b = float(prob[idx]); pr = int(pred[idx]); actual = int(y.iloc[idx])
    c1, c2, c3 = st.columns(3)
    card(c1, "Actual", CLASSES[actual]); card(c2, "Predicted", CLASSES[pr]); card(c3, "P(benign)", f"{p_b:.1%}")
    st.progress(min(max(p_b, 0.0), 1.0))
    st.markdown("✅ Correct prediction" if pr == actual else "⚠️ Model got this one wrong")
    est = model[-1] if hasattr(model, "steps") else model
    imp = None
    if hasattr(est, "coef_"): imp = np.abs(est.coef_.ravel())
    elif hasattr(est, "feature_importances_"): imp = est.feature_importances_
    if imp is not None:
        st.markdown(f"**Top 10 features driving {choice}**")
        s = pd.Series(imp, index=expected).sort_values().tail(10)
        st.bar_chart(s, color=AMBER, horizontal=True)
    else:
        st.caption(f"{choice} has no built-in feature weights, so no importance chart is shown.")
    st.dataframe(X.iloc[[idx]].T.rename(columns={X.index[idx]: "value"}), use_container_width=True)

