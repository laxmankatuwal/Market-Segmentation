import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = 'final_model.pkl'
try:
    loaded_model = pickle.load(open(filename, 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'final_model.sav' exists.")
    st.stop()

df = pd.read_csv("Clustered_Customer_Data.csv")

st.markdown(
    '''
    <style>
        body {background-color: Blue; color: white;}
        h1, h2, h3 {text-align: center; color: white;}
        .stNumberInput label {font-weight: bold; color: white;}
        .stButton button {background-color: #0e76a8; color: white;}
    </style>
    ''', 
    unsafe_allow_html=True
)

st.title("Market Segmentation")

with st.form("customer_form"):
    st.subheader("Enter Customer Details:")
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='OneOff Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Transactions', step=1)
    purchases_trx = st.number_input(label='Purchases Transactions', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='Percentage Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)
    submitted = st.form_submit_button("Submit")

if submitted:
    input_data = [[
        balance, balance_frequency, purchases, oneoff_purchases,
        installments_purchases, cash_advance, purchases_frequency,
        oneoff_purchases_frequency, purchases_installment_frequency,
        cash_advance_frequency, cash_advance_trx, purchases_trx,
        credit_limit, payments, minimum_payments, prc_full_payment, tenure
    ]]
    try:
        cluster = loaded_model.predict(input_data)[0]
        st.success(f"Data belongs to Cluster {cluster}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()
    
    cluster_data = df[df['Cluster'] == cluster]
    
    st.subheader(f"Cluster {cluster} Analysis")
    if not cluster_data.empty:
        for column in cluster_data.drop(['Cluster'], axis=1):
            plt.figure(figsize=(8, 4))
            sns.histplot(cluster_data[column], kde=True, bins=20, color="blue")
            plt.title(f"Distribution of {column} in Cluster {cluster}", color="white")
            plt.xlabel(column, color="white")
            plt.ylabel("Frequency", color="white")
            st.pyplot(plt.gcf())
            plt.close()
    else:
        st.warning(f"No data available for Cluster {cluster}.")
