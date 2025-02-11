import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# ---------------------
# Data Loading Section
# ---------------------
def load_data():
    uploaded_file = st.file_uploader("Upload MBA Decision Dataset (CSV)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")

            # Display dataset summary statistics and a peek
            st.subheader("Dataset Overview")
            st.dataframe(df.head())
            st.write(df.describe())  # Basic descriptive stats
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    return None

# ------------------------
# Preprocessing Section
# ------------------------
def preprocess_data(df):
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Check required columns
    required_columns = ['age', 'gender', 'undergraduate_major', 'undergraduate_gpa',
                       'years_of_work_experience', 'decided_to_pursue_mba?']

    # Check if required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None

    # Handle missing values
    df = df.dropna(subset=['decided_to_pursue_mba?'])

    # Convert categorical columns
    categorical_cols = ['gender', 'undergraduate_major', 'current_job_title',
                       'desired_post-mba_role', 'location_preference_(post-mba)']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Encode target variable
    le = LabelEncoder()
    df['decided_to_pursue_mba?'] = le.fit_transform(df['decided_to_pursue_mba?'])

    # Select features (excluding gre_gmat_score if missing)
    features = ['age', 'undergraduate_gpa', 'years_of_work_experience']

    # Add gre_gmat_score if it exists
    if 'gre_gmat_score' in df.columns:
        features.append('gre_gmat_score')
    else:
        st.warning("GRE/GMAT Score column not found. Proceeding without it.")

    # Add categorical features
    features.extend(['gender', 'undergraduate_major'])

    X = df[features]
    y = df['decided_to_pursue_mba?']

    return X, y, le

# -------------------------
# Visualization Section
# -------------------------
def show_visualizations(df):
    st.header("Data Visualizations")

    # Focus on the most directly relevant visualizations for initial insight
    st.subheader("Distribution of Key Variables")
    col1, col2 = st.columns(2)  # Display plots in two columns

    with col1:
        fig_age = px.histogram(df, x='age', title='Age Distribution')
        st.plotly_chart(fig_age)
        fig_gender = px.pie(df, names='gender', title='Gender Proportion')
        st.plotly_chart(fig_gender)  

    with col2:
        fig_gpa = px.histogram(df, x='undergraduate_gpa', title='GPA Distribution')
        st.plotly_chart(fig_gpa)

        mba_decision_counts = df['decided_to_pursue_mba?'].value_counts()
        fig_mba = px.bar(x=mba_decision_counts.index, y=mba_decision_counts.values, labels={'x': 'MBA Decision', 'y': 'Count'},
                        title='MBA Decision Distribution')  
        st.plotly_chart(fig_mba)


    st.subheader("Relationship with MBA Decision")
    col3, col4 = st.columns(2)
    with col3:
         fig_decision_age = px.box(df, x = 'decided_to_pursue_mba?', y= 'age', color = 'decided_to_pursue_mba?', title = "MBA vs Age")
         st.plotly_chart(fig_decision_age)
         contingency_table = pd.crosstab(df['gender'], df['decided_to_pursue_mba?'])
         fig_side_by_side = px.bar(contingency_table, x=contingency_table.index, y=contingency_table.columns, barmode= 'group', title='Gender vs. MBA Decision')
         st.plotly_chart(fig_side_by_side)

    with col4:
         fig_decision_experience = px.box(df, x = 'decided_to_pursue_mba?', y= 'years_of_work_experience', color = 'decided_to_pursue_mba?', title = "MBA vs Experience")
         st.plotly_chart(fig_decision_experience)
         fig_decision_gpa = px.box(df, x = 'decided_to_pursue_mba?', y= 'undergraduate_gpa', color = 'decided_to_pursue_mba?', title = "MBA vs GPA")
         st.plotly_chart(fig_decision_gpa)
# -------------------------
# Model Training Section
# -------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    with st.spinner("Training model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True,
                   labels=dict(x="Predicted", y="Actual"),
                   x=['No', 'Yes'], y=['No', 'Yes'])
    st.plotly_chart(fig)

    # Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    st.subheader("Feature Importance")
    st.dataframe(feature_importance_df)

    return model

# ----------------------------
# Prediction Interface
# ----------------------------
def prediction_interface(model, le):
    st.header("MBA Predictor")

    age = st.number_input("Age", min_value=20, max_value=50, value=25)
    gpa = st.number_input("Undergraduate GPA", min_value=2.0, max_value=4.0, value=3.0, step=0.1)  # Number Input with step
    experience = st.number_input("Years of Work Experience", min_value=0, max_value=15, value=3)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    major = st.selectbox("Undergraduate Major", ["Business", "Engineering", "Arts", "Science", "Economics"])

    # Convert categorical inputs
    gender_code = 0 if gender == "Male" else 1 if gender == "Female" else 2
    major_code = ["Business", "Engineering", "Arts", "Science", "Economics"].index(major)

    # Input Validation
    if not 2.0 <= gpa <= 4.0:
        st.error("GPA must be between 2.0 and 4.0.")
        return

    # Prepare input data
    input_data = [[age, gpa, experience, gender_code, major_code]]

    if st.button("Predict"):
        prediction = model.predict(input_data)
        result = le.inverse_transform(prediction)
        st.success(f"Prediction: {'Will Pursue MBA' if result[0] == 'Yes' else 'Will Not Pursue MBA'}")

# ----------------------------
# Main Application Flow
# ----------------------------
def main():
    st.title("MBA Predictor")

    # Add section for documentation links
    st.sidebar.header("Documentation")
    st.sidebar.markdown("[Dataset Source](https://www.kaggle.com/datasets/ashaychoudhary/dataset-mba-decision-after-bachelors)")

    # Data loading
    df = load_data()

    if df is not None:
        # Preprocessing
        processed_data = preprocess_data(df)
        if processed_data:
            X, y, le = processed_data

            # Visualization
            show_visualizations(df)

            # Model training
            st.header("Model Training")
            model = train_model(X, y)

            # Save model for predictions
            joblib.dump(model, 'mba_predictor_model.pkl')

            # Prediction interface
            prediction_interface(model, le)

if __name__ == "__main__":
    main()