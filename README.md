# Credit Card Fraud and Overdraft Detection

Detecting credit card fraud with data analytics and machine learning to identify suspicious patterns and improve decision-making.

## Dataset

[Credit Card Fraud Detection (Kaggle, mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 European cardholder transactions over two days in September 2013, of which only 492 (~0.17%) are fraud.

| Column | Meaning |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset (0 to ~172,800, about 48 hours) |
| `V1`–`V28` | Anonymized PCA components |
| `Amount` | Transaction amount |
| `Class` | Label: 1 = fraud, 0 = normal |

The CSV is expected at `data/creditcard.csv` (not committed to git). Download it from Kaggle, or with `kagglehub` (set `KAGGLE_USERNAME`/`KAGGLE_KEY` or place `kaggle.json` in `~/.kaggle/`).

## Project structure

```
data/creditcard.csv                    # local dataset (git-ignored)
notebooks/01_download_data.ipynb       # EDA, feature engineering, model training and evaluation
scripts/train.py                       # command-line version of the training pipeline
models/                                # saved models (git-ignored)
output/                                # saved charts (PNG)
requirements.txt
```

## Setup

Use Python 3.13 (the packages must match the interpreter that created the virtual environment).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name fraud-venv --display-name "Python (fraud)"
```

## Usage

- **Notebook:** open `notebooks/01_download_data.ipynb`, select the "Python (fraud)" kernel and run all cells. It loads the local CSV, engineers features, runs exploratory analysis, trains the models, evaluates them and saves charts to `output/` and models to `models/`.
- **Script:** `python scripts/train.py` trains and evaluates the same models from the command line. Use `--data <path>` and `--out <dir>` to override the defaults.

## Engineered features

| Feature | Definition |
|---|---|
| `small_spend` | 1 if `Amount` <= 10, else 0 |
| `large_spend` | 1 if `Amount` >= 100, else 0 |
| `hour` | Approximate hour of day, `(Time // 3600) % 24` (hour 0 is the dataset start, not a confirmed clock time) |

## Models

Stratified 80/20 train/test split (`random_state=42`); both models handle the class imbalance with class weights.

- **Logistic regression** (standardized inputs, `class_weight="balanced"`)
- **Random forest** (200 trees, `class_weight="balanced_subsample"`)

### Results (test set, 98 frauds, threshold 0.5)

| Model | ROC AUC | PR AUC | Frauds caught | False alarms | Best F1 (threshold) |
|---|---|---|---|---|---|
| Logistic regression | 0.974 | 0.726 | 89 | ~1,500 | 0.83 (very high threshold) |
| Random forest | 0.952 | 0.867 | 73 | 3 | 0.88 (~0.31) |

The random forest is the better model overall. Logistic regression catches slightly more fraud but at the cost of many false alarms. PR AUC is the more meaningful metric here because of the heavy class imbalance.

## Analysis and findings

Charts are saved in `output/`.

### Exploratory analysis (`eda_overview.png`)
- Fraud is extremely rare (~0.17%), so accuracy alone is misleading.
- Fraud rate by spend bucket: small purchases (<= 10) ~0.25%, large (>= 100) ~0.23%, mid-range ~0.09%.
- Statistical tests show both spend flags are associated with fraud, though the effects are small:

| Test | Result |
|---|---|
| Fraud rate, `small_spend` | 0.248% when 1 vs 0.132% when 0 (p ≈ 1e-12) |
| Fraud rate, `large_spend` | 0.227% when 1 vs 0.159% when 0 (p ≈ 6e-4) |
| Mann-Whitney U on `Amount` | p ≈ 9e-6 |

### Correlation with fraud (`correlation_with_class.png`)
V17, V14, V12 and V10 are the most negatively correlated with fraud (about -0.33 to -0.22); V11 and V4 are the most positive (about +0.15 and +0.13). No single feature is strongly correlated, so detection needs a combination of features.

### Hour of day (`hour_of_day.png`)
Transaction volume is lowest overnight (hours ~2-5) and steady from ~9 to ~22. The fraud rate is the reverse: ~1.7% at hour 2 and ~1.0% at hour 4, versus roughly 0.05-0.2% during the day. Fraud is concentrated when few normal transactions occur.

### Transaction amount by class (`amount_boxplot_by_class.png`)
The boxplot compares `log1p(Amount)` for normal and fraud transactions to make the central distributions readable despite the right-skewed amounts. Individual outliers are hidden in the plot; the accompanying table reports the 25th, 50th, 75th, 95th and 99th percentiles using all transactions.

### Fraud across two PCA features (`v14_v10_fraud_scatter.png`)
The scatter plot shows every transaction across `V14` and `V10`, fading normal points and highlighting fraud. It illustrates how these important anonymized features relate to the labels, while making clear that overlap means the pair alone is not a complete detection rule.

### Precision-recall curves (`precision_recall_curves.png`)
The random forest holds higher precision across most recall levels (PR AUC ~0.87 vs ~0.73), so it is the better model overall.

### Confusion matrices (`confusion_matrices.png`)
At threshold 0.5, logistic regression catches 89 of 98 frauds but raises ~1,500 false alarms; the random forest raises only 3 false alarms but misses 25 frauds (catches 73). It is a trade-off between catching more fraud and bothering fewer customers.

### Feature importance (`feature_importance.png`)
V14 is by far the most important feature for the random forest (~0.16), followed by V10, V12, V4 and V17, which match the most correlated features. The engineered features add little: `hour` ~0.004, `small_spend` ~0.0015, `large_spend` ~0.0007. Adding `hour` and the spend flags changed model metrics by only a point or two, probably because V1–V28 already carry most of that signal.

### Threshold analysis (`threshold_analysis.png`)
Raising the threshold increases precision and lowers recall. The random forest reaches its best F1 (~0.88) near a threshold of 0.31, better than at the default 0.5. Logistic regression's best F1 (~0.83) needs a very high threshold, showing its probabilities are poorly calibrated because of class weighting.

### Top-K fraud capture (`top_k_fraud_capture.png`)
This chart compares the share of test-set frauds captured when reviewing only the highest-scored transactions. At a 0.1% alert budget (57 transactions), the random forest captures 55 of 98 frauds (56.1%), versus 46 (46.9%) for logistic regression. At a 0.5% budget (285 transactions), they capture 88 (89.8%) and 87 (88.8%), respectively.

### Probability calibration (`calibration_curves.png`)
The calibration curve compares predicted probabilities with observed fraud frequencies in quantile bins; points nearer the diagonal are better calibrated. On this test split, the Brier scores are 0.000409 for the random forest, 0.023899 for logistic regression, and 0.001717 for a prevalence-only baseline (lower is better). Interpret Brier scores alongside the precision-recall and alert-capacity results because fraud is rare.

> The numbers above come from the current data, split and models. Update them if any of those change.

## Possible next steps

Stratified cross-validation, cost-based threshold selection using `Amount`, gradient boosting (XGBoost/LightGBM), Isolation Forest, and SMOTE or undersampling compared against class weights.
