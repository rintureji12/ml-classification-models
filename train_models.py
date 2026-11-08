"""Train 5 classifiers on the UCI Breast Cancer Wisconsin (Diagnostic) dataset,
save the models, the test split and the evaluation metrics."""
import os, warnings, joblib, pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# Load the UCI dataset directly with pandas
# Source: UCI ML Repository - Breast Cancer Wisconsin (Diagnostic), dataset id 17
# Raw wdbc.data has NO header: col 0 = ID, col 1 = diagnosis (M/B), cols 2-31 = 30 features.
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
ZIP_URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
base = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
        "concavity", "concave_points", "symmetry", "fractal_dimension"]
cols = ["id", "diagnosis"] + [f"{b}{i}" for i in (1, 2, 3) for b in base]  # mean, se, worst

def load_uci():
    try:                                   # 1) direct URL
        return pd.read_csv(URL, header=None, names=cols)
    except Exception as e:
        print("Direct URL failed:", e)
    try:                                   # 2) official zip download
        import zipfile, io, urllib.request
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ZIP_URL).read())) as z:
            return pd.read_csv(z.open("wdbc.data"), header=None, names=cols)
    except Exception as e:
        print("Zip URL failed:", e)
    if os.path.exists("wdbc.data"):        # 3) manually downloaded file next to this script
        return pd.read_csv("wdbc.data", header=None, names=cols)
    raise SystemExit("Could not load data. Download wdbc.data from the UCI page and place it here.")

df = load_uci()
df["target"] = df["diagnosis"].map({"M": 0, "B": 1})   # 0 = malignant, 1 = benign
df = df.drop(columns=[c for c in ("id", "diagnosis") if c in df.columns])
df.to_csv("full_dataset.csv", index=False)
X, y = df.drop(columns="target"), df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN":                 make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes":         GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
}
rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib")
    p, prob = m.predict(X_te), m.predict_proba(X_te)[:, 1]
    rows.append({"ML Model Name": name,
                 "Accuracy": accuracy_score(y_te, p), "AUC": roc_auc_score(y_te, prob),
                 "Precision": precision_score(y_te, p), "Recall": recall_score(y_te, p),
                 "F1": f1_score(y_te, p), "MCC": matthews_corrcoef(y_te, p)})
res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_summary.csv", index=False)
print(res.to_string(index=False))