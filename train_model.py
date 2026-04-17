import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("sample_data.csv")
X = df[["voltage","current","power"]]

model = IsolationForest(contamination=0.05)
model.fit(X)

joblib.dump(model, "model.pkl")
