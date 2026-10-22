import pandas as pd

# Load the data
df = pd.read_csv("churn.csv")

# First look
print(df.shape)
print(df.head())
print(df.info())

# Clean: TotalCharges has blank spaces stored as text
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())
df = df.dropna()

# How many customers churned?
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)