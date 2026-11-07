"""Streamlit evaluation app. It loads saved pipelines and never retrains them."""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "is_canceled"
POSITIVE_LABEL = 1
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}


@st.cache_data
def load_features():
    return json.loads((ROOT / "feature_columns.json").read_text(encoding="utf-8"))


st.set_page_config(page_title="Hotel Booking Cancellation Evaluation", layout="wide")
st.title("Hotel Booking Cancellation: Model Evaluation")
st.write(
    "Upload a labeled CSV containing the hotel booking input features and an `is_canceled` target column. "
    "The app loads a saved model and evaluates it; it does not retrain on the uploaded file."
)
st.caption("Target: whether the booking was canceled (1) or not canceled (0). This is an academic demonstration.")

uploaded = st.file_uploader("Upload labeled test data (.csv)", type=["csv"])
if uploaded is not None:
    data = pd.read_csv(uploaded)
    required = load_features()
    missing = [column for column in required + [TARGET] if column not in data.columns]
    if missing:
        st.error("CSV is missing required columns: " + ", ".join(missing))
        st.stop()
    if data.empty:
        st.error("The uploaded CSV contains no rows.")
        st.stop()

    X_test = data[required].copy()
    y_test = pd.to_numeric(data[TARGET], errors="coerce")
    if y_test.isna().any() or not set(y_test.unique()).issubset({0, 1}):
        st.error("The `is_canceled` target must contain only 0/1 labels.")
        st.stop()
    models = load_models()
    selected_name = st.selectbox("Choose a saved model", list(MODEL_FILES))
    model = models[selected_name]
    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(POSITIVE_LABEL)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    scores = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test == POSITIVE_LABEL, probabilities) if y_test.nunique() == 2 else float("nan"),
        "Precision": precision_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "F1": f1_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    st.subheader(f"{selected_name} results")
    st.dataframe(pd.DataFrame([scores]).style.format("{:.4f}"), use_container_width=True)
    st.subheader("Confusion matrix")
    st.write("Rows are actual labels; columns are predicted labels. Order: 0 (not canceled), 1 (canceled).")
    st.dataframe(
        pd.DataFrame(
            confusion_matrix(y_test, predictions, labels=[0, 1]),
            index=["Actual 0 (not canceled)", "Actual 1 (canceled)"],
            columns=["Predicted 0 (not canceled)", "Predicted 1 (canceled)"],
        ),
        use_container_width=True,
    )
    st.subheader("Classification report")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    st.download_button(
        "Download predictions",
        pd.DataFrame({"actual_is_canceled": y_test, "predicted_is_canceled": predictions}).to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )
