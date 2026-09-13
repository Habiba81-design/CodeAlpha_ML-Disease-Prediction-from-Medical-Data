# CodeAlpha_DiseasePrediction

Machine Learning internship project — CodeAlpha (Task 4: Disease Prediction from Medical Data).

## Problem

Predict whether a patient has heart disease based on clinical measurements —
age, chest pain type, cholesterol, ECG results, and related features. This is
a binary classification problem trained on the Cleveland subset of the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(304 patients).

## Project structure

```
CodeAlpha_DiseasePrediction/
├── 04_train_heart_model.py     # Cleaning, preprocessing, training, evaluation
├── requirements.txt
├── README.md
├── heart_disease_cleveland.csv # Cleveland-only subset (not committed — see Data)
└── outputs/
    └── heart_model/             # trained model, metrics, plots
```

## Data

The full combined UCI Heart Disease dataset (920 rows across 4 hospitals:
Cleveland, Hungary, Switzerland, VA Long Beach) is available on Kaggle by
searching **"UCI Heart Disease Data"**. This project uses only the
**Cleveland subset** (304 rows), filtered from the full file with:

```python
df = pd.read_csv("heart_disease_uci.csv")
cleveland = df[df["dataset"] == "Cleveland"]
```

Cleveland was chosen over the full combined dataset because the other three
hospitals are missing the `ca` and `thal` columns for 80-99% of their
patients — too much to responsibly impute. Cleveland itself has only 1-5
missing values across the whole dataset.

## Approach

1. **Cleaning** — dropped `id` (row identifier, no signal) and `dataset`
   (constant after filtering to Cleveland); converted boolean columns
   (`fbs`, `exang`) to 0/1 integers; binarized the target from a 5-level
   severity score (`num`, 0-4) into a simple `has_disease` (0/1) flag,
   matching the brief's "predict the possibility of disease" framing.
2. **Preprocessing** — numeric features (age, blood pressure, cholesterol,
   etc.) median-imputed and scaled; categorical features (chest pain type,
   ECG results, etc.) most-frequent-imputed and one-hot encoded; boolean
   features passed through unchanged. All handled in a single
   `ColumnTransformer` to keep training and future predictions consistent.
3. **Modeling** — Logistic Regression, Random Forest, and
   HistGradientBoosting, each tuned via `RandomizedSearchCV` (5-fold,
   scored on ROC-AUC), with `class_weight="balanced"`.
4. **Evaluation** — ROC-AUC, PR-AUC, precision, recall, F1, and confusion
   matrix on a held-out 20% validation split.

## Results

| Model | CV ROC-AUC | Validation ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Logistic Regression (best)** | 0.8993 | **0.9610** | 0.839 | 0.929 | 0.881 |
| Random Forest | 0.9068 | 0.9491 | 0.867 | 0.929 | 0.897 |
| HistGradientBoosting | 0.8858 | 0.9448 | 0.857 | 0.857 | 0.857 |

Best model: **Logistic Regression**, selected by highest validation ROC-AUC (0.961).

**Interpretation:** ROC-AUC of 0.96 reflects how strongly established
clinical predictors (chest pain type, ST depression, max heart rate) relate
to heart disease. Logistic Regression outperforming the more complex
ensemble models is notable — on a small (304-row), clean, roughly balanced
dataset with largely linear relationships between features and diagnosis, a
simple, interpretable model can beat more complex ones. Recall of 0.929
(catching 26 of 28 disease cases in validation) is the most clinically
relevant metric here, since a missed diagnosis is more costly than a false
alarm that gets ruled out by further testing.

## How to run

```bash
pip install -r requirements.txt
python3 04_train_heart_model.py
```

## Notes

- The target was binarized (disease present vs. absent) rather than kept as
  the original 5-level severity score, to match the brief and allow direct
  comparison with Task 1's evaluation approach.
- `xgboost` was intentionally not used to keep dependencies light;
  `HistGradientBoostingClassifier` is scikit-learn's native alternative.
