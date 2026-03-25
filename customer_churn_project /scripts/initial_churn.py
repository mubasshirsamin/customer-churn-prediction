import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset has loaded successfully.\n")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print("Shape:", df.shape)

print("\nColumn names:")
print("Columns:", df.columns.tolist())

print("\nData types and non-null counts:")
print(df.info())

print("\nMissing values present in each column:")
print(df.isnull().sum())

print("\nSummary statistics for numerical columns:")
print(df.describe())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(df["Churn"].value_counts(normalize=True) * 100)