# Credit-Card-Fraud-and-Overdraft-Detection
Detect credit card fraud and overdraft using data analytics can help identify suspicious patterns and improve decision-making.

## Data
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) and place it in `data/`.

## Usage
```
pip install -r requirements.txt
python src/fraud_detection.py --data data/creditcard.csv
pytest
```
The script summarizes class imbalance, trains a class-weighted random forest, and reports precision/recall and ROC AUC.
Note: the dataset has no account-balance fields, so overdraft detection would need additional data.
