import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the heart dataset
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\asus\Downloads\Disease.csv')

# Train the model
def train_model(data):
    # Preprocessing
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# Function to preprocess input data
def preprocess_input(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Create DataFrame from input data
    new_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal],   
    })
    
    # Use StandardScaler to scale the input data
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)
    
    return new_data_scaled

# Streamlit App
def main():
    st.title("Disease Predictor")
    st.write("This app predicts the likelihood of disease based on input parameters.")
    st.write("Made in SRM ")
    # Load data
    data = load_data()
    
    # Train model
    model, acc = train_model(data)
    st.write(f"Model Accuracy: {acc * 100:.0f}%")

    
    # Sidebar Inputs
    age = st.sidebar.number_input("Enter your Age", min_value=1)
    sex = st.sidebar.selectbox("Male or Female", ["Male", "Female"])
    cp = st.sidebar.number_input("Enter Value of CP", min_value=0)
    trestbps = st.sidebar.number_input("Enter Value of trestbps")
    chol = st.sidebar.number_input("Enter Value of chol")
    fbs = st.sidebar.number_input("Enter Value of fbs", min_value=0, max_value=1)
    restecg = st.sidebar.number_input("Enter Value of restecg")
    thalach = st.sidebar.number_input("Enter Value of thalach")
    exang = st.sidebar.number_input("Enter Value of exang", min_value=0, max_value=1)
    oldpeak = st.sidebar.number_input("Enter Value of oldpeak", min_value=0.0)
    slope = st.sidebar.number_input("Enter Value of slope")
    ca = st.sidebar.number_input("Enter Value of ca")
    thal = st.sidebar.number_input("Enter Value of thal")
    
    # Preprocess input data
    input_data = preprocess_input(age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    
    # Predict
    if st.sidebar.button("Predict"):
        result = model.predict(input_data)
        if result[0] == 0:
            st.write("No Heart Disease")
        else:
            st.write("Possibility of Heart Disease")

if __name__ == "__main__":
    main()
