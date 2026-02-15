# ML Assignment 2 - Classification Models with Streamlit

## a. Problem Statement
The objective of this assignment is to implement multiple classification models on the Wine Quality dataset, evaluate them using standard metrics, and deploy an interactive Streamlit app. The app must allow dataset upload, model selection, and display of evaluation metrics along with a confusion matrix or classification report.

---

## b. Dataset Description
- **Dataset**: Wine Quality (UCI Repository)
- **Instances**: ~4,900
- **Features**: 12 physicochemical properties (e.g., acidity, sugar, pH, alcohol)
- **Target**: Wine quality (converted to binary classification: good vs bad, where quality ≥ 6 is considered good)

---

## c. Models Used
Six classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN) Classifier  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison Table of Evaluation Metrics

| ML Model Name           | Accuracy | AUC     | Precision | Recall   | F1          | MCC     |
|------------------------|-----------|--------|------------|---------|--------|---------|
| Logistic Regression   | 0.7406     | 0.819   | 0.7857      | 0.7374  | 0.7608| 0.4793 |
| Decision Tree              | 0.7188      | 0.717   | 0.7572      | 0.7318   | 0.7443| 0.4323 |
| KNN                               | 0.7063     | 0.7737| 0.7202     | 0.7765  | 0.7473| 0.3994 |
| Naive Bayes                 | 0.7344     | 0.7927| 0.7582     | 0.7709  | 0.7645| 0.4600 |
| Random Forest           | 0.7969     | 0.8882| 0.8202     | 0.8156    | 0.8179 | 0.5883 |
| XGBoost                       | 0.8125      | 0.8787| 0.8362     | 0.8268   | 0.8315 | 0.6203 |

---

## d. Observations on Model Performance

| ML Model Name                        | Observation about model performance                                                                               |
|--------------------------------|-------------------------------------------------------------------------------------------|
| Logistic Regression                | Balanced precision and recall, good baseline model with stable performance.            |
| Decision Tree                           | Prone to overfitting, lower MCC compared to ensemble methods.                                  |
| KNN                                            | Performs reasonably well after scaling, but slightly weaker MCC.                                  |
| Naive Bayes                              | Assumptions about feature distributions limit performance, though recall is decent. |
| Random Forest (Ensemble)   | Strong performance due to ensemble averaging, high accuracy and MCC.                    |
| XGBoost (Ensemble)               | Best overall performance across all metrics, robust and consistent results.              |

---

## Deployment
- Streamlit App deployed on Streamlit Community Cloud.  
- [Live App Link](https://project-folder-5jcq5wwclnvx72gdxamwrz.streamlit.app/)  
