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
        # Display prediction result
        if predi == 1:
            st.write("The customer is eligible for a loan.")
            if age < 30:
                st.write("Recommendation: Offer a loan with a shorter duration to match the customer's young age.")
            elif age >= 30 and Age < 50:
                st.write("Recommendation: Provide a standard loan package with moderate terms.")
            else:
                st.write("Recommendation: Consider offering longer loan durations to align with the customer's senior age group.")
            
            if selected_saving_accounts == 'little':
                st.write("Recommendation: Encourage the customer to save more to build a stronger financial profile.")
            elif selected_saving_accounts == 'moderate':
                st.write("Recommendation: Suggest considering higher savings to increase financial stability.")
            elif selected_saving_accounts == 'rich':
                st.write("Recommendation: Acknowledge the customer's strong financial position and offer premium loan options.")
            
            if selected_checking_account == 'little':
                st.write("Recommendation: Recommend the customer to manage their checking account balance more effectively.")
            elif selected_checking_account == 'moderate':
                st.write("Recommendation: Maintain a steady balance in the checking account for better financial management.")
            elif selected_checking_account == 'rich':
                st.write("Recommendation: Utilize excess funds in the checking account to explore investment opportunities.")
            
            if credit_amount < 5000:
                st.write("Recommendation: Offer smaller loan amounts to match the customer's current credit needs.")
            elif credit_amount >= 5000 and Credit_amount < 10000:
                st.write("Recommendation: Provide a standard loan amount suitable for typical expenses.")
            else:
                st.write("Recommendation: Consider offering larger loan amounts to accommodate significant financial requirements.")
            
        elif predi == 0:
            st.write("The customer is not eligible for a loan.")
            if selected_job == 'unskilled':
                st.write("Recommendation: Advise the customer to seek employment opportunities with higher income potential.")
            elif selected_job == 'skilled':
                st.write("Recommendation: Explore alternative financing options or loan programs tailored for skilled workers.")
            else:
                st.write("Recommendation: Encourage the customer to improve their employment status for better financial stability.")
            
            if selected_saving_accounts == 'little' or selected_checking_account == 'little':
                st.write("Recommendation: Recommend focusing on building savings and improving financial management habits.")
            elif selected_saving_accounts == 'moderate' or selected_checking_account == 'moderate':
                st.write("Recommendation: Suggest maintaining a consistent savings and checking account balance to enhance financial stability.")
            elif selected_saving_accounts == 'rich' or selected_checking_account == 'rich':
                st.write("Recommendation: Acknowledge the customer's strong financial position and advise exploring alternative financing options.")
            
            if credit_amount > 20000:
                st.write("Recommendation: Consider reducing the loan amount or exploring alternative financing options to match the customer's financial capacity.")
            else:
                st.write("Recommendation: Review the loan amount and duration to ensure alignment with the customer's financial situation.")
            
        else:
            st.warning("Unable to make a prediction. Please investigate further.")
            st.write("Recommendation: Conduct a detailed assessment of the customer's financial situation and explore personalized solutions.")
