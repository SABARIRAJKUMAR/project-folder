import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Inject Bootstrap CSS
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="text-primary text-center">ML Classification Models Demo</h1>', unsafe_allow_html=True)

# a. Dataset upload option
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown('<h3 class="text-success">Data Preview</h3>', unsafe_allow_html=True)
    st.write(df.head())

    # b. Model selection dropdown
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    if "quality" not in df.columns:
        st.error("Dataset must contain a 'quality' column as target.")
    else:
        X = df.drop("quality", axis=1)
        y = (df["quality"] >= 6).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Select model
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "XGBoost":
            model = xgb.XGBClassifier(eval_metric="logloss")

        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        # c. Display evaluation metrics
        st.markdown('<h3 class="text-info">Evaluation Metrics</h3>', unsafe_allow_html=True)
        st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        st.write("AUC:", round(roc_auc_score(y_test, y_prob), 4))
        st.write("Precision:", round(precision_score(y_test, y_pred), 4))
        st.write("Recall:", round(recall_score(y_test, y_pred), 4))
        st.write("F1:", round(f1_score(y_test, y_pred), 4))
        st.write("MCC:", round(matthews_corrcoef(y_test, y_pred), 4))

        # d. Confusion matrix or classification report
        st.markdown('<h3 class="text-danger">Confusion Matrix</h3>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown('<h3 class="text-warning">Classification Report</h3>', unsafe_allow_html=True)
        st.text(classification_report(y_test, y_pred))
