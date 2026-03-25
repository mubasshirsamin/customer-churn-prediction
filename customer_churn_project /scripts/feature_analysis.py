import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

training_df = pd.read_csv("../output/train_data.csv")

# This step separates target and feature variables
x_train = training_df.drop(columns=["Churn"])
y_train = training_df["Churn"]

# Building the same Logistic Regression pipeline earlier
# used in the model training file
logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000, random_state=42))
])

# Training the model using the training dataset
logistic_pipeline.fit(x_train, y_train)

print("\nModel trained successfully for interpretation")

# Extracting the feature names
feature_variables = x_train.columns

# Extracting the coefficients found from building the model
coefficients = logistic_pipeline.named_steps["model"].coef_[0]

# This data frame shows each feature along with its coefficients
coefficient_df = pd.DataFrame({
    "Feature": feature_variables,
    "Coefficients": coefficients
})

# Sorting features by their coefficient value
coefficient_df = coefficient_df.sort_values(by="Coefficients", ascending=False)

print("\nFeatures increasing churn risk in order")
print((coefficient_df.head(10)))

print("\nFeatures reducing churn risk in order")
print(coefficient_df.tail(10))

coefficient_df.to_csv("../output/logistic_regression_coefficients.csv", index=False)