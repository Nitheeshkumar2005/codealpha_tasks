import pandas as pd

# Load dataset
df = pd.read_csv("dataset/credit_risk_dataset.csv")

print("=" * 50)
print("CREDIT SCORING MODEL - DATA ANALYSIS")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

# Remove missing values
df = df.dropna()

print("\nShape After Cleaning:")
print(df.shape)

# Feature Engineering

df["income_loan_ratio"] = df["person_income"] / df["loan_amnt"]

df["loan_per_income"] = df["loan_amnt"] / df["person_income"]

print("\nFeature Engineering Completed")

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

print("\nCategorical Columns Encoded Successfully")

print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDECISION TREE RESULTS")
print("Accuracy :", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall   :", recall_score(y_test, dt_pred))
print("F1 Score :", f1_score(y_test, dt_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))

from sklearn.metrics import roc_auc_score

lr_prob = lr_model.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, lr_prob)

print("\nROC-AUC Score:", roc_score)

import pickle

pickle.dump(
    rf_model,
    open("models/credit_score_model.pkl", "wb")
)

print("Model Saved Successfully")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix")
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, rf_pred))


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=False)
plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_matrix.png")
plt.show()

from sklearn.metrics import roc_curve

rf_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("images/roc_curve.png")
plt.show()