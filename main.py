import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

@st.cache_data
def load_and_preprocess_data():
    file_path = r"428412_1_En_25_MOESM1_ESM.csv"
    data = pd.read_csv(file_path)
    date_columns = ['dob', 'policy_start_dt', 'policy_end_dt']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    data.dropna(subset=['dob'], inplace=True)
    data['dob_year'] = data['dob'].dt.year
    data['dob_month'] = data['dob'].dt.month
    data['dob_day'] = data['dob'].dt.day
    data['policy_duration'] = (data['policy_end_dt'] - data['policy_start_dt']).dt.days
    data['policy_age'] = (pd.Timestamp.now() - data['policy_start_dt']).dt.days
    data.drop(columns=date_columns, inplace=True)
    categorical_columns = data.select_dtypes(include=['object']).columns.difference(date_columns)
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    return data

data = load_and_preprocess_data()
X = data.drop(columns=['fraud', 'policy_ref', 'member_id'])
y = data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Train Neural Network model
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn_model.fit(X_train_scaled, y_train)

# Data visualization
st.title("Fraud Detection Model Ideal Insurance Inc.")
st.header("Data Overview")
st.write(data.head())
st.header("Data Distribution")
for col in data.columns:
    st.subheader(f"{col}")
    fig, ax = plt.subplots()
    sns.histplot(data[col], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# Streamlit interface
st.sidebar.header("Input Features")

# Input fields
inputs = {}
for col in X.columns:
    inputs[col] = st.sidebar.number_input(f"{col}")

# Button to make predictions
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "Neural Network (MLP)"])
if st.sidebar.button("Enter"):
    input_data = pd.DataFrame([inputs])
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)
        st.write(f"Model: Random Forest")
    elif model_choice == "XGBoost":
        prediction = xgb_model.predict(input_data)
        st.write(f"Model: XGBoost")
    else:
        input_data_scaled = scaler.transform(input_data)
        prediction = nn_model.predict(input_data_scaled)
        st.write(f"Model: Neural Network (MLP)")
    st.write(f"Prediction: {'Fraudulent' if prediction[0] == 1 else 'Genuine'}")
