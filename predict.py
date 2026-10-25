
import sys

import joblib
import pandas as pd

bundle = joblib.load("outputs/fraud_model.joblib")
df = pd.read_csv(sys.argv[1]).drop(columns=["Class"], errors="ignore")
proba = bundle["model"].predict_proba(df)[:, 1]
df["fraud_probability"] = proba
df["is_fraud"] = (proba >= bundle["threshold"]).astype(int)
df.to_csv("predictions.csv", index=False)
print(df["is_fraud"].value_counts())
