import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# Création de l'API
app = FastAPI(title="Churn Prediction API")


# Charger le modèle et le preprocessing
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# Structure des données reçues
class Customer(BaseModel):
    tenure: int
    monthly_charges: float
    support_calls: int
    contract: str


# Endpoint de prédiction
@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame([customer.model_dump()])

    data_transformed = preprocessor.transform(data)

    prediction = model.predict(data_transformed)

    return {
        "churn": int(prediction[0])
    }