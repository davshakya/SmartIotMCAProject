from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib, numpy as np

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
    return "ANOMALY" if model.predict(X)[0] == -1 else "NORMAL"

@app.post("/data")
def receive(item: dict):
    item["prediction"] = predict(item)
    DATA.append(item)
    return {"ok": True}

@app.get("/data")
def get_data():
    return DATA[-50:]
