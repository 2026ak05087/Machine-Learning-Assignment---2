# Rice Yield Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify agricultural records
into High Yield and Low Yield categories using ICRISAT district-level
agricultural data. Five classification algorithms are implemented
and compared using standard evaluation metrics.

## 2. Dataset Description

- Dataset: ICRISAT District-Level Data
- Original records: 16,146
- Original columns: 80
- Records retained after yield filtering: 14,981
- Target column: Yield_Class
- Class 0: Low Yield
- Class 1: High Yield
- Yield threshold: 1,417.1 kg/ha
- Training records: 11,984
- Testing records: 2,997

Records with non-positive rice yield were excluded. Rice area,
rice production, and rice yield were removed from the input features
to prevent target leakage.

## 3. GitHub Repository Link

Add your GitHub repository URL here after creating the repository.

## 4. Models Used and Evaluation Results

The following results were obtained from the test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8969 | 0.9627 | 0.8961 | 0.8979 | 0.8970 | 0.7938 |
| Decision Tree | 0.8809 | 0.8809 | 0.8816 | 0.8798 | 0.8807 | 0.7618 |
| KNN | 0.8976 | 0.9528 | 0.8994 | 0.8952 | 0.8973 | 0.7951 |
| Naive Bayes | 0.8001 | 0.8310 | 0.7684 | 0.8591 | 0.8112 | 0.6045 |
| Support Vector Machine | 0.9009 | 0.9644 | 0.8979 | 0.9045 | 0.9012 | 0.8018 |

## 5. Observations

### Logistic Regression
Achieved 89.69% accuracy and an AUC score of 0.9627,
indicating strong classification performance.

### Decision Tree
Achieved 88.09% accuracy. Its performance was lower than
Logistic Regression, KNN, and SVM in this experiment.

### K-Nearest Neighbors
Achieved 89.76% accuracy and 0.8994 precision.
It performed competitively with Logistic Regression.

### Naive Bayes
Achieved 80.01% accuracy and 0.8310 AUC.
It had the lowest accuracy among the five models, but its
recall was 0.8591.

### Support Vector Machine
Achieved the highest accuracy (90.09%), AUC (0.9644),
F1 score (0.9012), and MCC (0.8018) among the five models.

### Overall Winner
Support Vector Machine was the best-performing model in
this experiment based on the reported test metrics.

## 6. Project Files

- app.py: Streamlit application
- requirements.txt: Python dependencies
- test_data.csv: Test dataset with target column
- model/: Saved classification pipelines
- dense_transformer.py: Dense conversion transformer
- model_comparison_results.csv: Model evaluation results

## 7. How to Run

Install dependencies:

    pip install -r requirements.txt

Run the Streamlit application:

    streamlit run app.py

Upload a test CSV containing the required input features and
Yield_Class target column. Select a model to view the evaluation
metrics, confusion matrix, and classification report.
