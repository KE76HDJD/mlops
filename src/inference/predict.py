import joblib
import pandas as pd


# Charger le preprocessing
preprocessor = joblib.load("models/preprocessor.pkl")

# Charger le modèle
model = joblib.load("models/model.pkl")


# Nouveau client
new_customer = pd.DataFrame([
    {
        "tenure": 10,
        "monthly_charges": 80.0,
        "support_calls": 5,
        "contract": "monthly"
    }
])


# Transformer les données
new_customer_transformed = preprocessor.transform(new_customer)


# Faire la prédiction
prediction = model.predict(new_customer_transformed)


print("Prediction :", prediction)