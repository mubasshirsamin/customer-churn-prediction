import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("../output/cleaned_telco_churn.csv")

# Plot for overall churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Overall Churn distribution")
plt.xlabel("Churn")
plt.ylabel("Customer Number")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.savefig("../output/overall_churn_distribution.png")
plt.show()

# Plot for Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by contract type")
plt.xlabel("Contract type")
plt.ylabel("Customer number")
plt.tight_layout()
plt.savefig("../output/churn_by_contract_type.png")
plt.show()

# Plot for Monthly charges by Churn status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("MonthlyCharges($)")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.savefig("../output/monthly_charges_by_churn.png")
plt.show()

# Plot for Churn by Internet Service
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="InternetService", hue="Churn")
plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("../output/churn_by_internet_service.png")
plt.show()

# Plot for Churn by Payment Method
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn")
plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("../output/churn_by_payment_method.png")
plt.show()

# Plot for Churn by Tech Support
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="TechSupport", hue="Churn")
plt.title("Churn by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("../output/churn_by_tech_support.png")
plt.show()

# Plot for Churn by Online Security
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="OnlineSecurity", hue="Churn")
plt.title("Churn by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("../output/churn_by_online_security.png")
plt.show()

# Plot for Churn by Paperless Billing
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="PaperlessBilling", hue="Churn")
plt.title("Churn by Paperless Billing")
plt.xlabel("Paperless Billing")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("../output/churn_by_paperless_billing.png")
plt.show()

# Plot for Churn by Senior Citizen
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="SeniorCitizen", hue="Churn")
plt.title("Churn by Senior Citizen Status")
plt.xlabel("Senior Citizen")
plt.ylabel("Number of Customers")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.savefig("../output/churn_by_senior_citizen.png")
plt.show()

# Plot for Tenure by Churn Status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Tenure by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.savefig("../output/tenure_by_churn_status.png")
plt.show()

# Plot for Total Charges by Churn Status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="TotalCharges")
plt.title("Total Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Total Charges($)")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.savefig("../output/total_charges_by_churn_status.png")
plt.show()