from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

model = pickle.load(open("model/churn_model.pkl", "rb"))

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

class CustomerFeatures(BaseModel):
    call_failure: int
    complains: int
    subscription_length: int
    charge_amount: int
    seconds_of_use: int
    frequency_of_use: int
    frequency_of_sms: int
    distinct_called_numbers: int
    age_group: int
    tariff_plan: int
    status: int
    age: int
    customer_value: float

@app.post("/predict/")
def predict_churn(features: CustomerFeatures):
    feature_list = [
        features.call_failure,
        features.complains,
        features.subscription_length,
        features.charge_amount,
        features.seconds_of_use,
        features.frequency_of_use,
        features.frequency_of_sms,
        features.distinct_called_numbers,
        features.age_group,
        features.tariff_plan,
        features.status,
        features.age,
        features.customer_value
    ]
    
    prediction = model.predict(np.array(feature_list).reshape(1, -1))
    
    return {"Churn Prediction": str(prediction[0])}
