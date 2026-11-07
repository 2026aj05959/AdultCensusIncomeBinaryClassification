"""Download Adult Census Income, train five classifiers, and save artifacts."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "income"
POSITIVE_LABEL = ">50K"
RANDOM_STATE = 42
SAMPLE_SIZE = 10_000  # Keeps local training and KNN evaluation manageable.


def clean_data():
    """Fetch the UCI Adult dataset and normalize missing values and labels."""
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y_frame = adult.data.targets.copy()
    y = y_frame.iloc[:, 0].astype(str).str.strip().str.replace(r"\.$", "", regex=True)

    for column in X.select_dtypes(include=["object", "category"]).columns:
        X[column] = X[column].astype("string").str.strip().replace("?", pd.NA)

    # Use a stratified 10,000-row subset for reasonable laptop and free-tier use.
    if len(X) > SAMPLE_SIZE:
        X, _, y, _ = train_test_split(
            X, y, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    return X.reset_index(drop=True), y.reset_index(drop=True)


def make_pipeline(model, numeric_features, categorical_features):
    # Dense one-hot output supports GaussianNB. The 10,000-row subset keeps
    # memory use manageable; with 14 source features it remains practical.
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", model)])


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(POSITIVE_LABEL)
    positive_probabilities = model.predict_proba(X_test)[:, positive_index]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test == POSITIVE_LABEL, positive_probabilities),
        "Precision": precision_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main():
    X, y = clean_data()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    results = []
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator, numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        scores = evaluate(pipeline, X_test, y_test)
        results.append({"Model": name, **scores})
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)
        print(f"\n{name}\n" + "\n".join(f"  {k}: {v:.4f}" for k, v in scores.items()))
        print("  Confusion matrix (rows=true, columns=predicted; labels <=50K, >50K):")
        print(confusion_matrix(y_test, pipeline.predict(X_test), labels=["<=50K", ">50K"]))

    # The app requires the target column in the uploaded test CSV.
    test_data = X_test.copy()
    test_data[TARGET] = y_test.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    pd.DataFrame(results).to_csv(ROOT / "metrics.csv", index=False)
    (ROOT / "feature_columns.json").write_text(
        json.dumps(X.columns.tolist(), indent=2), encoding="utf-8"
    )
    print(f"\nSaved held-out test data and five models under: {ROOT}")


if __name__ == "__main__":
    main()
