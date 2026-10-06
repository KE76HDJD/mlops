import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Charger les données
df = pd.read_csv("data/raw/churn.csv")


# 2. Séparer les features et la target
X = df[[
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract"
]]

y = df["churn"]


# 3. Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Définir les colonnes numériques et catégorielles
numeric_features = [
    "tenure",
    "monthly_charges",
    "support_calls"
]

categorical_features = [
    "contract"
]


# 5. Créer le preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# 6. Apprendre le preprocessing sur TRAIN
X_train_transformed = preprocessor.fit_transform(X_train)


# 7. Appliquer le preprocessing sur TEST
X_test_transformed = preprocessor.transform(X_test)


# 8. Créer le modèle
model = LogisticRegression()


# 9. Entraîner le modèle
model.fit(X_train_transformed, y_train)
joblib.dump(model, "models/model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

# 10. Faire les prédictions
predictions = model.predict(X_test_transformed)

# 11. Calculer les métriques
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)


# 12. Afficher les résultats
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

print("\nPredictions :", predictions)
print("Actual      :", y_test.to_numpy())
print("Confusion Matrix :\n", cm)

print("\nAccuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-score  :", f1)