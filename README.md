# Breast Cancer Classification – 5 ML Models + Streamlit App

## a. Problem Statement
Predict whether a breast tumour is **malignant (0)** or **benign (1)** from 30 numeric
features computed from digitised images of a fine-needle aspirate (FNA) of a breast mass.
This is a **binary classification** problem. Five classifiers (Logistic Regression, Decision Tree,
KNN, Gaussian Naive Bayes, SVM) are trained on the same data, compared with six metrics and
served through an interactive Streamlit app.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic), dataset id 17:
  https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
  (Wolberg, Mangasarian, Street & Street, 1993; DOI 10.24432/C5DW2B)
- **Instances:** 569 (min. required 500) | **Features:** 30 (min. required 12) | **Target:** `target` (0 = malignant: 212, 1 = benign: 357)
- **Features:** mean, standard error and "worst" value of 10 cell-nucleus measurements: radius, texture,
  perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.
- **Missing values:** none.
- **Split:** 80/20 stratified (`random_state=42`) → 455 train / 114 test. The 114-row test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` is part of the saved pipeline for Logistic Regression, KNN and SVM
  (Decision Tree and Naive Bayes do not need scaling).

## c. GitHub Repository Link
`https://github.com/rintureji12/ml-classification-models`

Streamlit app: `https://ml-classification-models-rintureji.streamlit.app/`

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

(Precision / Recall / F1 are for the positive class = benign, evaluated on the 114-row test set.)

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint best (98.25% accuracy, highest AUC 0.9954, MCC 0.962). The classes are close to linearly separable once features are standardised, so a simple, interpretable linear model is enough. |
| Decision Tree | Weakest model (92.1% accuracy, AUC 0.916). A single depth-5 tree is high-variance and overfits the training set; its hard leaf probabilities also give a lower AUC. Ensembles would likely help. |
| KNN | Good (95.6% accuracy, recall 0.972) but behind the linear/kernel models. It depends on scaling and on the choice of k, and is sensitive to the 30-dimensional feature space. |
| Naive Bayes | 93.9% accuracy but a high AUC (0.988): ranking quality is strong, while the independence assumption is violated by highly correlated features (radius/perimeter/area), hurting hard-label accuracy and MCC. |
| Support Vector Machine | Ties Logistic Regression on accuracy, precision, recall, F1 and MCC (AUC 0.995). The RBF kernel with scaled features handles the boundary well, at slightly higher training cost. |
| **Overall winner** | **Logistic Regression** – it ties SVM on every threshold-based metric, has the highest AUC, and is simpler, faster and more interpretable. |

> Note: with only 114 test rows, one misclassified sample changes accuracy by ~0.9%, so the top two models
> are statistically indistinguishable.

## Repository Structure
```
├── app.py                 # Streamlit app
├── train_models.py        # trains & saves the models, creates test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv          # 114-row test split (features + target) for upload
├── full_dataset.csv       # full dataset for reference
└── model/                 # saved .joblib pipelines + metrics_summary.csv
```

## Streamlit App Features
1. CSV upload (test data with all 30 features + `target`)
2. Model-selection dropdown (5 models)
3. Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
4. Confusion matrix and classification report
5. Comparison table of all 5 models on the uploaded data (models are loaded, never retrained)

     ### Additional features
     1. ROC curves for all five models on one plot.
     2. A confidence histogram for actual malignant vs benign samples.
     3. A leaderboard chart that ranks the models by any metric.
     4. A case explorer for inspecting a single test row.
     5.  Top 10 feature importance for Logistic Regression and the Decision Tree.
     6. Downloads for the predictions CSV and the sample test_data.csv.

## Run Locally
```
pip install -r requirements.txt
python train_models.py      # optional, models are already saved
streamlit run app.py
```
