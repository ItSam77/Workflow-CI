from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")  

class PredictRequest(BaseModel):
    Age: int
    EnvironmentSatisfaction: int
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MaritalStatus: int
    OverTime: int
    StockOptionLevel: int
    TotalWorkingYears: int
    YearsAtCompany: int

@app.post("/predict")
def predict(req: PredictRequest):
    # Convert the request to a pandas DataFrame with the correct column order
    feature_data = {
        'Age': req.Age,
        'EnvironmentSatisfaction': req.EnvironmentSatisfaction,
        'JobInvolvement': req.JobInvolvement,
        'JobLevel': req.JobLevel,
        'JobSatisfaction': req.JobSatisfaction,
        'MaritalStatus': req.MaritalStatus,
        'OverTime': req.OverTime,
        'StockOptionLevel': req.StockOptionLevel,
        'TotalWorkingYears': req.TotalWorkingYears,
        'YearsAtCompany': req.YearsAtCompany
    }
    
    # Create DataFrame and convert to numpy array
    df = pd.DataFrame([feature_data])
    arr = df.values
    
    pred = model.predict(arr)
    return {"prediction": int(pred[0])}
