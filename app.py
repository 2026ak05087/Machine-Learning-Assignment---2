
import streamlit as st
import pandas as pd
import joblib
import os

# Import the custom transformer used by the Naive Bayes pipeline
from dense_transformer import DenseTransformer

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


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

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


# --------------------------------------------------
# 2. MODEL DIRECTORY AND MODEL FILES
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


# --------------------------------------------------
# 3. UPLOAD TEST DATASET
# --------------------------------------------------

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

        st.subheader("Dataset Preview")

        st.dataframe(test_data.head())

        # Check target column

        if "Yield_Class" not in test_data.columns:

            st.error(
                "The uploaded CSV must contain "
                "the Yield_Class target column."
            )

            st.stop()

        # Separate features and target

        y_test = test_data["Yield_Class"].astype(int)

        X_test = test_data.drop(columns=["Yield_Class"])

        # Validate target values

        if not y_test.isin([0, 1]).all():

            st.error(
                "Yield_Class must contain only 0 and 1."
            )

            st.stop()

        if y_test.nunique() < 2:

            st.error(
                "The test dataset must contain both classes (0 and 1) "
                "to calculate AUC."
            )

            st.stop()


        # --------------------------------------------------
        # 4. SELECT CLASSIFICATION MODEL
        # --------------------------------------------------

        st.header("2. Select Classification Model")

        selected_model = st.selectbox(
            "Choose a model",
            list(model_files.keys())
        )

        model_path = os.path.join(
            MODEL_DIR,
            model_files[selected_model]
        )

        if not os.path.isfile(model_path):

            st.error(
                "Saved model file not found: " + model_path
            )

            st.info(
                "Please upload all five model files "
                "inside the model folder in your GitHub repository."
            )

            st.stop()


        # --------------------------------------------------
        # 5. EVALUATE MODEL
        # --------------------------------------------------

        if st.button("Evaluate Model"):

            try:

                # Load saved model pipeline

                pipeline = joblib.load(model_path)

                # Predict test data

                y_pred = pipeline.predict(X_test)

                # Calculate positive-class probabilities

                y_prob = pipeline.predict_proba(X_test)[:, 1]

                # Evaluation metrics

                accuracy = accuracy_score(y_test, y_pred)

                auc = roc_auc_score(y_test, y_prob)

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                mcc = matthews_corrcoef(
                    y_test,
                    y_pred
                )


                # --------------------------------------------------
                # 6. DISPLAY EVALUATION METRICS
                # --------------------------------------------------

                st.header("3. Evaluation Metrics")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Accuracy",
                    f"{accuracy:.4f}"
                )

                col2.metric(
                    "AUC Score",
                    f"{auc:.4f}"
                )

                col3.metric(
                    "Precision",
                    f"{precision:.4f}"
                )

                col4, col5, col6 = st.columns(3)

                col4.metric(
                    "Recall",
                    f"{recall:.4f}"
                )

                col5.metric(
                    "F1 Score",
                    f"{f1:.4f}"
                )

                col6.metric(
                    "MCC Score",
                    f"{mcc:.4f}"
                )


                # --------------------------------------------------
                # 7. CONFUSION MATRIX
                # --------------------------------------------------

                st.header("4. Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=[0, 1]
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        "Actual Low Yield (0)",
                        "Actual High Yield (1)"
                    ],
                    columns=[
                        "Predicted Low Yield (0)",
                        "Predicted High Yield (1)"
                    ]
                )

                st.dataframe(cm_df)


                # --------------------------------------------------
                # 8. CLASSIFICATION REPORT
                # --------------------------------------------------

                st.header("5. Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                    target_names=[
                        "Low Yield",
                        "High Yield"
                    ],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(
                    report
                ).transpose().round(4)

                st.dataframe(report_df)

                st.success(
                    selected_model + " evaluated successfully!"
                )


            except Exception as error:

                st.error(
                    "Model evaluation failed. "
                    "Please check the saved model and input columns."
                )

                st.exception(error)


    except Exception as error:

        st.error(
            "Could not read the uploaded CSV file."
        )

        st.exception(error)


else:

    st.info(
        "Please upload your test_data.csv file to begin."
    )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.caption(
    "Machine Learning Assignment 2 | ICRISAT Dataset"
)
