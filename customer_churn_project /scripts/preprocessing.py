import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../output/cleaned_telco_churn.csv")

print(f"Shape of the dataset before preprocessing: {df.shape}")

# This step removes customerID as it does not hold predictive information
# to determine whether a customer will churn or not
df = df.drop(columns=["customerID"])

# X contains all the driving factors for churn (input variables)
# Y only contains the target variable
x = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Feature matrix shape: {x.shape}")
print(f"Target vector shape: {y.shape}")

# This step converts text categories into numeric dummy variables
# so that they can be later worked directly with machine learning models

x_encoded = pd.get_dummies(x, drop_first=True)

print("\nSuccessfully encoded categorical variables")
print(f"Shape of the encoded feature matrix: {x_encoded.shape}")

# Keeping 80% of the data for the training split and using the other
# 20% for the test split, and the training data will be later used
# with n-fold cross-validation during model training

X_train, X_test, Y_train, Y_test = train_test_split(
    x_encoded,
    y,
    test_size = 0.20,
    random_state = 42,
# stratification helps keep the similar churn ratio in both
# train and test sets
    stratify = y
)

print("\nAfter Train-test split.")
print(f"Input variables training set shape: {X_train.shape}")
print(f"Input variables test set shape: {X_test.shape}")
print(f"Target variable training set shape: {Y_train.shape}")
print(f"Target variable test set shape: {Y_test.shape}")

# This step creates a full training dataset by combining the training features
# with the training target column
final_training_df = X_train.copy()
final_training_df["Churn"] = Y_train.values

# This step creates a full testing dataset by combining the training features
# with the testing target column
final_testing_df = X_test.copy()
final_testing_df["Churn"] = Y_test.values

# Saving final processed training and testing data set to output file
final_training_df.to_csv("../output/train_data.csv", index=False)
final_testing_df.to_csv("../output/test_data.csv", index=False)