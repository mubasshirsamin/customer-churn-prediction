import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pre-split training and testing datasets
training_df = pd.read_csv("../output/train_data.csv")
test_df = pd.read_csv("../output/test_data.csv")

print(f"Shape of training dataset: {training_df.shape}")
print(f"Shape of testing dataset: {test_df.shape}")

# These steps separates target and feature variables
# in both training and test datasets
x_train = training_df.drop(columns=["Churn"])
y_train = training_df["Churn"]

x_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

# This dictionary evaluates multiple models in one loop
models = {
    "Logistic Regression": Pipeline([
    # StandardScaler scaled the numeric range of the features
        ("scaler", StandardScaler()),
    # This step gives the algorithm enough iterations to finish
    # training the dataset sufficiently
        ("model", LogisticRegression(max_iter=3000, random_state=42))
    ]),
    # This model captures non-linear patterns and max_depth reduces overfitting
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    # This model combines many trees to improve prediction performance
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}
for model_name, model in models.items():
    print("\n" + "=" * 50)
    print(f"{model_name}")
    print("=" * 50)

    # Perform 5-fold cross-validation to prepare the model better on
    # unseen data and reduce the risk of overfitting
    cross_validation = cross_val_score(
        model,
        x_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )
    print("\nCross-validation ROC-AUC scores:")
    print(cross_validation)
    print(f"Average cross-validation ROC-AUC: {cross_validation.mean():.4f}")

    # Training the model on the entire training data set
    model.fit(x_train, y_train)

    print(f"\nSuccessfully trained {model_name}.")

    # Training the model again before final evaluation
    model.fit(x_train, y_train)

    # Making predictions on the test set
    y_prediction = model.predict(x_test)
    probability_of_y_prediction = model.predict_proba(x_test)[:, 1]

    # Evaluating the accuracy of the model
    accuracy = accuracy_score(y_test, y_prediction)
    roc_auc = roc_auc_score(y_test, probability_of_y_prediction)

    # This step evaluates the model using accuracy and ROC-AUC
    # accuracy represents overall correctness, while AUC-ROC
    # measures how well the model separates churners and non-churners
    print(f"\nTest performance of the {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # The confusion matrix demonstrates correct and incorrect classification
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_prediction))

    print("\nClassification Report:")
    print(classification_report(y_test, y_prediction))
