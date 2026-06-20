import pandas as pd
from sklearn.preprocessing import LabelEncoder


# loading the dataset


df = pd.read_csv("dataset/heart_disease_uci.csv")



print("=" * 60)
print("HEART DISEASE PREDICTION - DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())


# this is for dataset infomation...


print("\nDataset Information:")
df.info()


# to see Missig values analyse


print("\nMissing Values:")
print(df.isnull().sum())

#target column 


print("\nTarget Column Distribution:")
print(df["num"].value_counts())

print("\nTarget Column Percentage:")
print(
    round(
        df["num"].value_counts(normalize=True) * 100,
        2
    )
)

#converting target to binary

df["num"] = df["num"].apply(
    lambda value: 0 if value == 0 else 1
)

print("\nBinary Target Distribution:")
print(df["num"].value_counts())

print("\nBinary Target Percentage:")
print(
    round(
        df["num"].value_counts(normalize=True) * 100,
        2
    )
)

#feature analysis
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_columns)

print("\nCategorical Features:")
print(categorical_columns)

print("\nAnalysis Completed Successfully")

# Data cleaning 
print("\nStarting Data Cleaning...")

# Remove ID column
df.drop("id", axis=1, inplace=True)

# Numerical Missing Values
numerical_columns = [
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"
]

for column in numerical_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


categorical_columns = [
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#Label encodeing

print("\nEncoding Categorical Features...")

label_encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(
        df[column]
    )

print("\nEncoding Completed Successfully")

print("\nEncoded Dataset Preview:")
print(df.head())

# features and targets

X = df.drop("num", axis=1)
y = df["num"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# train and split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Test Split Completed")

print(f"Training Data : {X_train.shape}")
print(f"Testing Data  : {X_test.shape}")

#Logisstic regression...

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression Accuracy:")
print(f"{logistic_accuracy:.4f}")

# Random forest classifier

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(
    X_test
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print("\nRandom Forest Accuracy:")
print(f"{random_forest_accuracy:.4f}")

# svm

from sklearn.svm import SVC

svm_model = SVC()

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

print("\nSVM Accuracy:")
print(f"{svm_accuracy:.4f}")

# model compare
results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "SVM"
        ],
        "Accuracy": [
            logistic_accuracy,
            random_forest_accuracy,
            svm_accuracy
        ]
    }
)

print("\nModel Comparison")
print(results.sort_values(
    by="Accuracy",
    ascending=False
))

from sklearn.metrics import classification_report

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        random_forest_predictions
    )
)

# confusion matrix
from sklearn.metrics import confusion_matrix

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    random_forest_predictions
)

print(cm)

# save the model

import joblib

joblib.dump(
    random_forest_model,
    "models/heart_disease_model.pkl"
)

print("\nModel Saved Successfully")