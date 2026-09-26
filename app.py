
import streamlit as st  # Used to create the interactive web interface
import pandas as pd  # Used to create and manage transaction data
import joblib  # Used to load the saved model and scaler

# Load the saved model and scaler
model_path = "fraud_detection_model/final_model.pkl"
scaler_path = "fraud_detection_model/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define the exact feature order used during model training
final_features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_TRANSFER",
    "sender_balance_change",
    "destination_balance_change",
    "balance_error",
    "sender_balance_ratio",
    "destination_balance_ratio",
    "balance_difference"
]

# Page title and description
st.title("Fraud Detection System")
st.write("Enter transaction details to predict whether the transaction is fraudulent.")

# Transaction inputs
amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "CASH_IN", "CASH_OUT", "DEBIT", "TRANSFER"]
)

oldbalance_org = st.number_input("Old Sender Balance", min_value=0.0, value=10000.0)
newbalance_orig = st.number_input("New Sender Balance", min_value=0.0, value=5000.0)
oldbalance_dest = st.number_input("Old Destination Balance", min_value=0.0, value=20000.0)
newbalance_dest = st.number_input("New Destination Balance", min_value=0.0, value=25000.0)

# Create transaction dataframe
input_data = pd.DataFrame({
    "amount": [amount],
    "type": [transaction_type],
    "oldbalanceOrg": [oldbalance_org],
    "newbalanceOrig": [newbalance_orig],
    "oldbalanceDest": [oldbalance_dest],
    "newbalanceDest": [newbalance_dest]
})

# Feature engineering
input_data["sender_balance_change"] = (
    input_data["oldbalanceOrg"] - input_data["newbalanceOrig"]
)

input_data["destination_balance_change"] = (
    input_data["newbalanceDest"] - input_data["oldbalanceDest"]
)

input_data["balance_error"] = (
    input_data["amount"] - input_data["sender_balance_change"]
)

input_data["sender_balance_ratio"] = (
    input_data["amount"] / (input_data["oldbalanceOrg"] + 1)
)

input_data["destination_balance_ratio"] = (
    input_data["amount"] / (input_data["oldbalanceDest"] + 1)
)

input_data["balance_difference"] = (
    input_data["sender_balance_change"]
    - input_data["destination_balance_change"]
)

# One-hot encode transaction type
encoded_data = pd.get_dummies(
    input_data,
    columns=["type"],
    drop_first=False
)

# Ensure all training-time transaction type columns are present
for column in [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_TRANSFER"
]:
    if column not in encoded_data.columns:
        encoded_data[column] = 0

# Remove PAYMENT because it is the reference category
if "type_PAYMENT" in encoded_data.columns:
    encoded_data = encoded_data.drop(columns=["type_PAYMENT"])

# Match the exact feature order used during training
encoded_data = encoded_data.reindex(
    columns=final_features,
    fill_value=0
)

# Validate feature structure before prediction
if encoded_data.shape[1] != model.n_features_in_:
    st.error("Feature count mismatch. Prediction stopped.")
    st.stop()

if list(encoded_data.columns) != final_features:
    st.error("Feature order mismatch. Prediction stopped.")
    st.stop()

# Scale the prepared features
X_scaled = scaler.transform(encoded_data)

# Prediction
if st.button("Predict Transaction"):

    prediction = model.predict(X_scaled)[0]
    fraud_probability = model.predict_proba(X_scaled)[0, 1] * 100

    if prediction == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Transaction Appears Legitimate")

    st.metric(
        "Fraud Probability",
        f"{fraud_probability:.2f}%"
    )
