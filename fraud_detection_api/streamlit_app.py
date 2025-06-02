import streamlit as st
import requests
import json
import mysql.connector
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Initialize session state ---
if "form_values" not in st.session_state:
    st.session_state.form_values = {f"V{i}": 0.0 for i in range(1, 29)}
    st.session_state.form_values["Time"] = 0.0
    st.session_state.form_values["Amount"] = 0.0

st.title("💳 Credit Card Fraud Detection System")

# --- Connect to SQL and fetch random transaction ---
def get_random_transaction(is_fraud=False):
    db = mysql.connector.connect(
        user="root",
        password=os.getenv("SECURE_KEY"),
        database="creditcardfraud"
    )
    cursor = db.cursor(dictionary=True)
    query = "SELECT * FROM transactions WHERE Class = 1 ORDER BY RAND() LIMIT 1" if is_fraud else "SELECT * FROM transactions ORDER BY RAND() LIMIT 1"
    cursor.execute(query)
    transaction = cursor.fetchone()
    cursor.close()
    db.close()
    return transaction

def generate_sql_transaction(is_fraud=False):
    transaction = get_random_transaction(is_fraud)
    if transaction:
        st.session_state.form_values["Time"] = transaction["Time"]
        st.session_state.form_values["Amount"] = transaction["Amount"]
        for i in range(1, 29):
            st.session_state.form_values[f"V{i}"] = transaction[f"V{i}"]

# --- Auto-fill Buttons ---
col_a, col_b = st.columns(2)
with col_a:
    if st.button("✅ Auto-fill from Legitimate Transaction"):
        generate_sql_transaction()
with col_b:
    if st.button("⚠️ Auto-fill from Fraudulent Transaction"):
        generate_sql_transaction(is_fraud=True)

# --- Input UI ---
col1, col2 = st.columns(2)
with col1:
    time = st.number_input("Time", value=st.session_state.form_values["Time"], key="Time")
with col2:
    amount = st.number_input("Amount", value=st.session_state.form_values["Amount"], key="Amount")

input_features = {
    "Time": time,
    "Amount": amount
}
for i in range(1, 29, 2):
    col1, col2 = st.columns(2)
    with col1:
        input_features[f"V{i}"] = st.number_input(
            f"V{i}", value=st.session_state.form_values[f"V{i}"], key=f"V{i}", format="%.4f"
        )
    with col2:
        input_features[f"V{i+1}"] = st.number_input(
            f"V{i+1}", value=st.session_state.form_values[f"V{i+1}"], key=f"V{i+1}", format="%.4f"
        )

# --- Predict Button ---
if st.button("🔍 Predict"):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post("http://127.0.0.1:8000/predict", data=json.dumps(input_features), headers=headers)

        if response.status_code == 200:
            result = response.json()
            prediction_label = result['prediction']
            probability = result['probability']

            st.success(f"🎯 Prediction: {prediction_label}")
            st.write(f"🧪 Fraud Probability: `{probability:.5f}`")

            # Risk Bar
            st.progress(min(probability, 1.0))
            if probability >= 0.8:
                st.error("🚨 High Risk of Fraud") 
            elif probability >= 0.5:
                st.warning("⚠️ Medium Risk")
            else:
                st.info("✅ Low Risk")

            # Final message
            if prediction_label.strip().lower() == "fraud":
                st.markdown("🔎 The above transaction is **fraudulent** and the probability of it being a fraudulent transaction is mentioned above.")
            else:
                st.markdown("🔎 The above transaction is **legitimate** and the probability of it being a fraudulent transaction is mentioned above.")

            # Debug (optional)
            # st.write("🧾 Raw Prediction Response:", result)

        else:
            st.error("❌ Something went wrong.")
            st.write(response.text)

    except requests.exceptions.RequestException as e:
        st.error("🚫 Could not connect to backend.")
        st.exception(e)


# --- Documentation: How to use the app ---
with st.expander("📘 How to use this app"):
    st.markdown("""
    **Instructions to Use the Fraud Detection App**  
    1. **Start the FastAPI server** in your terminal using:
       ```
       uvicorn app:app --reload
       ```
    2. **Launch the Streamlit app** using:
       ```
       streamlit run st_app.py
       ```
    3. Use **Auto-fill buttons** to load real or fraudulent transaction data from your database.
    4. Optionally, manually change values using the form.
    5. Click **🔍 Predict** to check if the transaction is fraudulent.
    """)

# --- Show All Fraudulent Transactions ---
with st.expander("📄 Show All Fraudulent Transactions"):
    try:
        # Create SQLAlchemy engine
        engine = create_engine(f"mysql+mysqlconnector://root:{os.getenv('SECURE_KEY')}@localhost/creditcardfraud")

        fraud_query = "SELECT Time, Amount, Class FROM transactions WHERE Class = 1"
        fraud_df = pd.read_sql_query(fraud_query, engine)
        fraud_df = fraud_df.rename(columns={"Amount": "Amount ($)"})

        st.markdown(f"**🔢 Total Fraudulent Transactions:** `{len(fraud_df)}`")
        st.dataframe(fraud_df, use_container_width=True)
        st.markdown("Note: Class `1` = Fraudulent, Class `0` = Legitimate")

        # Download button
        csv_data = fraud_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Fraud Transactions as CSV",
            data=csv_data,
            file_name="fraud_transactions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ Failed to fetch fraudulent transactions from the database.")
        st.exception(e)
