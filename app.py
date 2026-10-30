
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ICRISAT Yield Classification",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Rice Yield Classification")
st.write(
    "Compare five machine learning classifiers "
    "using ICRISAT district-level agricultural data."
)

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing input features and Yield_Class",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully.")
        st.write("Dataset shape:", test_data.shape)
        st.dataframe(test_data.head())

        if "Yield_Class" not in test_data.columns:
            st.error(
                "The uploaded CSV must contain the Yield_Class target column."
            )
            st.stop()

        y_test = test_data["Yield_Class"].astype(int)
        X_test = test_data.drop(columns=["Yield_Class"])

        if not y_test.isin([0, 1]).all():
            st.error("Yield_Class must contain only 0 and 1.")
            st.stop()

        st.header("2. Select Classification Model")

        selected_model = st.selectbox(
            "Choose a model",
            list(model_files.keys())
        )

        model_path = os.path.join(
            MODEL_DIR,
            model_files[selected_model]
        )

        if not os.path.exists(model_path):
            st.error("Saved model file not found: " + model_path)
            st.stop()

        if st.button("Evaluate Model"):
            try:
                pipeline = joblib.load(model_path)

                # Predict directly from the original CSV columns
                y_pred = pipeline.predict(X_test)

                # Get positive-class probabilities for AUC
                y_prob = pipeline.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)
                precision = precision_score(
                    y_test, y_pred, zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, zero_division=0
                )
                f1 = f1_score(
                    y_test, y_pred, zero_division=0
                )
                mcc = matthews_corrcoef(y_test, y_pred)

                st.header("3. Evaluation Metrics")

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("Precision", f"{precision:.4f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", f"{recall:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")

                st.header("4. Confusion Matrix")

                cm = confusion_matrix(
                    y_test, y_pred, labels=[0, 1]
                )

                st.write(
                    pd.DataFrame(
                        cm,
                        index=["Actual Low Yield (0)",
                               "Actual High Yield (1)"],
                        columns=["Predicted Low Yield (0)",
                                 "Predicted High Yield (1)"]
                    )
                )

                st.header("5. Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=["Low Yield", "High Yield"],
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose().round(4)
                )

            except Exception as error:
                st.error(
                    "Model evaluation failed. Check the saved "
                    "pipeline and uploaded CSV columns."
                )
                st.exception(error)

    except Exception as error:
        st.error("Could not read the uploaded CSV file.")
        st.exception(error)

else:
    st.info("Please upload your test_data.csv file to begin.")

st.caption(
    "Machine Learning Assignment 2 | ICRISAT Dataset"
)
