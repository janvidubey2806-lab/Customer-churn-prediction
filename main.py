import pandas as pd

# Load the dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Display basic information
print("\nCUSTOMER CHURN DATASET")
print("----------------------")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())
# STEP 6: DATA CLEANING

print("\n--- DATA CLEANING ---")

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# Remove unnecessary customer ID column
df = df.drop(columns=["customerID"])

# Check cleaned dataset
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nData cleaning completed successfully!")
# STEP 7: EXPLORATORY DATA ANALYSIS

import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- CUSTOMER CHURN ANALYSIS ---")

# Count customers who churned
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Calculate churn percentage
churn_rate = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn Percentage:")
print(churn_rate)

# Create count plot
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.show()
# STEP 8: CHURN ANALYSIS BY CUSTOMER DETAILS

print("\n--- CHURN BY CONTRACT TYPE ---")

print(pd.crosstab(df["Contract"], df["Churn"]))

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
# STEP 9: PREPARE DATA FOR MACHINE LEARNING

from sklearn.model_selection import train_test_split

# Convert Churn into numbers
y = df["Churn"].map({"No": 0, "Yes": 1})

# Separate input features
X = df.drop("Churn", axis=1)

# Convert text columns into numeric columns
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- MACHINE LEARNING DATA PREPARATION ---")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
# STEP 10: TRAIN MACHINE LEARNING MODEL

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\n--- MODEL TRAINING COMPLETE ---")
print("Our Logistic Regression model is trained successfully!")
# STEP 11: MODEL EVALUATION

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# STEP 12: SAVE TRAINED MODEL

import joblib

# Save the model
joblib.dump(model, "customer_churn_model.pkl")

print("\n--- MODEL SAVED SUCCESSFULLY ---")
print("Model file: customer_churn_model.pkl")
# STEP 13: TEST CUSTOMER CHURN PREDICTION

sample_customer = X_test.iloc[[0]]

prediction = model.predict(sample_customer)

if prediction[0] == 1:
    print("\nCustomer is likely to leave the service.")
else:
    print("\nCustomer is likely to stay with the service.")

# STEP 14: SAVE MODEL WITH FEATURE COLUMNS

joblib.dump({
    "model": model,
    "columns": X.columns.tolist()
}, "customer_churn_model.pkl")

print("\nModel and feature columns saved successfully!")
