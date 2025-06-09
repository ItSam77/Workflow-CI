from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")  

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(req: PredictRequest):
    arr = np.array(req.features).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": int(pred[0])}
