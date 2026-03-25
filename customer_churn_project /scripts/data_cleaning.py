import pandas as pd
import numpy as np

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Successfully loaded raw dataset")
print(f"Initial shape of dataset: {df.shape}")

# This removes extra spaces and converts values to numeric
# invalid or empty entries will be converted to NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")

print("\nMissing values after cleaning TotalCharges")
print(df.isnull().sum())

# Median Imputation to fill in missing values of TotalCharges
total_charges_median = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_median)

print(f"\nFilled missing TotalCharges values after median imputation: {total_charges_median:.2f}")

# Checking the dataset for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}")

# This step removes duplicate rows if any are found
if duplicate_rows > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows were found")

# This step converts the churn column into numeric form later
# used for modelling
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("\nChurn distribution for mapping:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(df["Churn"].value_counts(normalize=True) * 100)

categorical_cols = df.select_dtypes(include=["object", "string"]).columns

print("\nGoing through categorical columns:")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(f"Unique values ({df[col].nunique()}):")
    print(df[col].unique())

print("\nFinal check for missing values:")
print(df.isnull().sum())

print("\nFinal data types:")
print(df.dtypes)

print(f"Final cleaned dataset: {df.shape}")

df.to_csv("../output/cleaned_telco_churn.csv", index=False)

print("\nCleaned dataset saved successfully to ../output/cleaned_telco_churn.csv")


