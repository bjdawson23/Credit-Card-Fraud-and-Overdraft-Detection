import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fraud_detection as fd


def synthetic(n=400):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["V1", "V2", "Amount"])
    df[fd.TARGET] = (df["V1"] > 1.5).astype(int)
    return df


def test_summarize():
    s = fd.summarize(synthetic())
    assert s["transactions"] == 400 and 0 < s["fraud_rate"] < 1


def test_train():
    _, report, auc = fd.train_and_evaluate(synthetic())
    assert auc > 0.9 and "precision" in report
