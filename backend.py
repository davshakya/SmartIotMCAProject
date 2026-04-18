from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
DATA = []


def predict(item):
    X = np.array([[item["voltage"], item["current"], item["power"]]])
    prediction = model.predict(X)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

    return prediction, confidence


@app.post("/data")
def receive(item: dict):
    prediction, confidence = predict(item)
    item["prediction"] = prediction
    item["detected_event"] = prediction
    if confidence is not None:
        item["confidence"] = round(confidence, 4)
    DATA.append(item)
    return {"ok": True}


@app.get("/data")
def get_data():
    return DATA[-50:]
