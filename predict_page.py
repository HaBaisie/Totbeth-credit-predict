import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open('credit_predict3.sav', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

gradient_boost = data["model"]
S_encoder = data["S_encoder"]
J_encoder = data["J_encoder"]
H_encoder = data["H_encoder"]
SA_encoder = data["SA_encoder"]
CA_encoder = data["CA_encoder"]
P_encoder = data["P_encoder"]
feature_names = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']

def show_predict_page():
    st.title("Software Developer Credit Risk Prediction")

    st.write("""### We need some information for prediction""")

    Sex = ("male", "female")
    Job = ("unskilled and non-resident", "unskilled and resident", "skilled", "highly skilled")
    Housing = ("own", "free", "rent")
    Saving_accounts = ("none", "little", "quite rich", "rich", "moderate")
    Checking_account = ("little", "moderate", "none", "rich")
    Purpose = ("radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others")

    selected_sex = st.selectbox("Sex", Sex)
    selected_job = st.selectbox("Skill Level", Job)
    selected_housing = st.selectbox("Housing", Housing)
    selected_saving_accounts = st.selectbox("Saving accounts rate", Saving_accounts)
    selected_checking_account = st.selectbox("Checking account Rate", Checking_account)
    selected_purpose = st.selectbox("Purpose of Credit", Purpose)

    age = st.number_input("Age", min_value=18, step=1)
    credit_amount = st.number_input("Credit amount in Naira", min_value=0, step=1)
    duration = st.number_input("Duration", min_value=0, step=1)

    ok = st.button("Predict")
    if ok:
        # Create a dictionary to map user input to feature names
        user_input = {
            "Age": age,
            "Sex": selected_sex,
            "Job": selected_job,
            "Housing": selected_housing,
            "Saving accounts": selected_saving_accounts,
            "Checking account": selected_checking_account,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Purpose": selected_purpose
        }

        # Convert user input to DataFrame
        X_df = pd.DataFrame([user_input], columns=feature_names)

        # Encode categorical features
        X_df["Sex"] = S_encoder.transform(X_df["Sex"])
        X_df["Job"] = J_encoder.transform(X_df["Job"])
        X_df["Housing"] = H_encoder.transform(X_df["Housing"])
        X_df["Saving accounts"] = SA_encoder.transform(X_df["Saving accounts"])
        X_df["Checking account"] = CA_encoder.transform(X_df["Checking account"])
        X_df["Purpose"] = P_encoder.transform(X_df["Purpose"])

        # Make prediction
        predi = gradient_boost.predict(X_df)

        # Display prediction result
        if predi == 1:
            st.write("The customer is eligible for a loan.")
            if income < 20000:
                st.write("Recommendation: Consider offering a lower loan amount or a longer repayment period.")
            elif income >= 20000 and income < 50000:
                st.write("Recommendation: Offer a standard loan package.")
            else:
                st.write("Recommendation: Offer premium loan packages with additional benefits.")
        elif predi == 0:
            st.write("The customer is not eligible for a loan.")
            if credit_score < 600:
                st.write("Recommendation: Provide guidance on improving credit score before reapplying for a loan.")
            elif credit_score >= 600 and credit_score < 700:
                st.write("Recommendation: Suggest alternative financing options with flexible terms.")
            else:
                st.write("Recommendation: Encourage the customer to build a stronger financial profile and reapply later.")
        else:
            st.warning("Unable to make a prediction. Please investigate further.")
            st.write("Recommendation: Conduct a thorough review of the customer's financial documents and history.")

