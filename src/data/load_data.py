import pandas as pd

df = pd.read_csv("data/raw/churn.csv")

print(df.head())
print("\nShape :", df.shape)
print("\nTypes :")
print(df.dtypes)
print("\nValeurs manquantes :")
print(df.isnull().sum())

X=df[["tenure","monthly_charges","support_calls","contract"]]
y=df["churn"]

print("\n : x")
print(X)

print("\n :y")

print(y)