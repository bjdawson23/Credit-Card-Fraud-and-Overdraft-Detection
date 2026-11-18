# Credit Card Fraud and Overdraft Detection

## Branton Dawson

## Capstone Project November 2026

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

```

## Engineered features

| Feature | Definition |
|---|---|
| `small_spend` | 1 if `Amount` <= 10, else 0 |
| `large_spend` | 1 if `Amount` >= 100, else 0 |
| `hour` | Approximate hour of day, `(Time // 3600) % 24` (hour 0 is the dataset start, not a confirmed clock time) |

### Correlation with fraud (`correlation_with_class.png`)
V17, V14, V12 and V10 are the most negatively correlated with fraud (about -0.33 to -0.22); V11 and V4 are the most positive (about +0.15 and +0.13). No single feature is strongly correlated, so detection needs a combination of features.

### Hour of day (`hour_of_day.png`)
Transaction volume is lowest overnight (hours ~2-5) and steady from ~9 to ~22. The fraud rate is the reverse: ~1.7% at hour 2 and ~1.0% at hour 4, versus roughly 0.05-0.2% during the day. Fraud is concentrated when few normal transactions occur.

### Fraud across two PCA features (`v14_v10_fraud_scatter.png`)
The scatter plot shows every transaction across `V14` and `V10`, fading normal points and highlighting fraud. It illustrates how these important anonymized features relate to the labels, while making clear that overlap means the pair alone is not a complete detection rule.

## Possible next steps

Testing wit the XGBoost model.
