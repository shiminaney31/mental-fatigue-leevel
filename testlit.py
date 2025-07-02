import streamlit as st
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset and encoder
df = pd.read_csv(r'C:\Users\HP\mental_fatigue_data.csv')
with open('workload_encoder.pkl', 'rb') as f:
    workload_encoder = pickle.load(f)

# Encode workload column
df['workload'] = workload_encoder.transform(df['workload'])

# Features and target
X = df.drop(columns=['fatigue_level'])
y = df['fatigue_level']
y_encoded = pd.factorize(y)[0]  # for consistent label format

# Streamlit page settings
st.set_page_config(page_title="Mental Fatigue Dashboard", layout="centered")
st.title("🧠 Mental Fatigue Level Predictor with Model Comparison")

# Sidebar inputs
st.sidebar.header("📊 Input Daily Habits")
sleep = st.sidebar.number_input("Sleep Hours", min_value=0.0, value=6.0)
screen_time = st.sidebar.number_input("Screen Time (hrs)", min_value=0.0, value=5.0)
workload = st.sidebar.selectbox("Academic Workload", ["light", "medium", "heavy"])
water = st.sidebar.number_input("Water Intake (cups)", min_value=0.0, value=6.0)
breaks = st.sidebar.number_input("Breaks Taken", min_value=0, value=2)

# Model selection
st.sidebar.header("🧠 Select Machine Learning Model")
model_choice = st.sidebar.selectbox("Choose Model", [
    "Logistic Regression", 
    "Decision Tree", 
    "K-Nearest Neighbors", 
    "Support Vector Machine (SVM)"
])

# Predict button
if st.sidebar.button("🔍 Predict Fatigue Level"):
    try:
        # Encode workload
        workload_encoded = workload_encoder.transform([workload])[0]
        input_data = [[sleep, screen_time, workload_encoded, water, breaks]]

        # Train selected model
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        elif model_choice == "Support Vector Machine (SVM)":
            model = SVC()
        else:
            st.error("Invalid model choice.")
            st.stop()

        model.fit(X, y_encoded)
        prediction = model.predict(input_data)[0]
        y_pred = model.predict(X)

        fatigue_labels = pd.factorize(y)[1]  # original labels
        predicted_label = fatigue_labels[prediction]

        # Tabs for Prediction and Evaluation
        tab1, tab2 = st.tabs(["📈 Prediction", "📊 Evaluation Metrics"])

        with tab1:
            st.subheader("📈 Prediction Result")
            st.success(f"Predicted Fatigue Level: **{predicted_label.upper()}** using {model_choice}")

            col1, col2, col3 = st.columns(3)
            col1.metric("🛌 Sleep (hrs)", sleep)
            col2.metric("📱 Screen Time", screen_time)
            col3.metric("🥤 Water Intake", water)
            col1.metric("📚 Workload", workload)
            col2.metric("⏸️ Breaks", breaks)

            chart_df = pd.DataFrame({
                'Metric': ['Sleep', 'Screen Time', 'Water', 'Breaks'],
                'Value': [sleep, screen_time, water, breaks]
            })
            st.subheader("📊 Habit Overview")
            st.bar_chart(chart_df.set_index("Metric"))

        with tab2:
            st.subheader("📊 Evaluation Metrics")
            accuracy = accuracy_score(y_encoded, y_pred)
            precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_encoded, y_pred)

            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**Precision:** {precision:.2f}")
            st.write(f"**Recall:** {recall:.2f}")
            st.write(f"**F1 Score:** {f1:.2f}")

            st.write("**Confusion Matrix:**")
            st.dataframe(pd.DataFrame(cm,
                                      columns=[f"Pred {c}" for c in fatigue_labels],
                                      index=[f"Actual {c}" for c in fatigue_labels]))

        st.info("Model trained on current dataset; result is for demonstration only.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
