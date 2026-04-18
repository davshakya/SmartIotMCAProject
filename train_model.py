import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42


def build_training_data():
    normal_df = pd.read_csv("sample_data.csv")[["voltage", "current", "power"]].copy()
    normal_df["target"] = "NORMAL"

    theft_df = normal_df.sample(n=len(normal_df), replace=True, random_state=RANDOM_STATE).copy()
    theft_df["power"] = theft_df["power"].sample(
        n=len(theft_df), replace=True, random_state=RANDOM_STATE + 1
    ).reset_index(drop=True)
    theft_df["power"] = theft_df["power"].clip(lower=0.0, upper=0.5)
    theft_df["target"] = "THEFT"

    fault_df = normal_df.sample(n=len(normal_df), replace=True, random_state=RANDOM_STATE + 2).copy()
    fault_df["power"] = fault_df["power"].sample(
        n=len(fault_df), replace=True, random_state=RANDOM_STATE + 3
    ).reset_index(drop=True)
    fault_df["power"] = fault_df["power"].clip(lower=20.0, upper=30.0)
    fault_df["target"] = "FAULT"

    training_df = pd.concat([normal_df, theft_df, fault_df], ignore_index=True)
    return training_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


training_df = build_training_data()
X = training_df[["voltage", "current", "power"]]
y = training_df["target"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=RANDOM_STATE,
)
model.fit(X, y)

joblib.dump(model, "model.pkl")
