import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# 1. Load and clean
df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)   # Yes = 1, No = 0

X = df.drop(columns=["Churn"])
y = df["Churn"]

# 2. Split: 80% to train, 20% to test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing: scale numbers, one-hot encode text categories
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# 4. Two models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    ),
}

results = {}
pipes = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    pipes[name] = pipe

# 5. Compare
print("\nMODEL COMPARISON")
print(pd.DataFrame(results).T.round(3))

# 6. Save the best model (by ROC-AUC)
best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
best_pipe = pipes[best_name]
joblib.dump(best_pipe, "churn_model.joblib")
print(f"\nBest model: {best_name} (saved as churn_model.joblib)")

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, best_pipe.predict(X_test)))

# 7. Top churn drivers from Logistic Regression
lr = pipes["Logistic Regression"]
names = lr.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
print("\nFactors that INCREASE churn most:")
print(coefs.tail(8).round(2))
print("\nFactors that REDUCE churn most:")
print(coefs.head(8).round(2))