"""
Week 4 - E-Governance Complaint Priority Prediction
Synthetic academic dataset; replace data.csv with an approved real dataset for real deployment.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv("data.csv")
X = df.drop(columns=["complaint_id", "priority"])
y = df["priority"]

categorical = ["category"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = RandomForestClassifier(
    n_estimators=250, max_depth=8, min_samples_leaf=3,
    random_state=42, class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, average="weighted", zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, pred, average="weighted", zero_division=0), 4))
print("F1 Score :", round(f1_score(y_test, pred, average="weighted", zero_division=0), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, pred, zero_division=0))

joblib.dump(pipeline, "complaint_priority_model.joblib")
print("\nModel saved as complaint_priority_model.joblib")
