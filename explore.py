import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def explore_data():
    df = pd.read_csv("german_credit_data.csv", index_col=0)

    # Fill missing values and map job categories
    df["Saving accounts"] = df["Saving accounts"].fillna("none")
    df["Checking account"] = df["Checking account"].fillna("none")
    df["Job"] = df["Job"].map({
        0: "unskilled and non-resident",
        1: "unskilled and resident",
        2: "skilled",
        3: "highly skilled"
    })
    st.title("German Credit Data Exploration")

    st.header("Data Overview")
    st.write("Displaying the first few rows of the dataset:")
    st.write(df.head())

    st.write("Summary statistics of the dataset:")
    st.write(df.describe())

    st.header("Data Visualization")

    st.subheader("Distribution of Age by Gender")
    fig_age_gender = px.histogram(df, x="Age", color="Sex", histnorm="percent", barmode="overlay",
                                  color_discrete_map={"male": "rgba(0, 87, 233, 0.6)", "female": "rgba(255, 0, 189, 0.6)"},
                                  facet_col="Sex", facet_col_wrap=2)
    fig_age_gender.update_layout(title="Distribution of Age by Gender", title_x=0.5, bargap=0.05)
    st.plotly_chart(fig_age_gender)

    st.subheader("Distribution of Purpose by Sex")
    fig_purpose_sex = px.histogram(df, x="Purpose", color="Sex", histnorm="percent", barmode="group", width=800)
    fig_purpose_sex.update_layout(title="Distribution of Purpose by Sex", title_x=0.5)
    st.plotly_chart(fig_purpose_sex)

    st.subheader("Credit amount by Purpose and Risk")
    fig_credit_purpose_risk = px.box(df, x="Purpose", y="Credit amount", color="Sex", width=900)
    fig_credit_purpose_risk.update_layout(title="Credit amount by Purpose and Risk", title_x=0.5)
    st.plotly_chart(fig_credit_purpose_risk)

    st.subheader("Credit amount and duration influence on credit risk")
    fig_box_credit_risk = plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Risk", y="Credit amount")
    plt.xlabel("Risk")
    plt.ylabel("Credit Amount")
    plt.title("Credit amount influence on credit risk")
    st.pyplot(fig_box_credit_risk)

    fig_box_duration_risk = plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Risk", y="Duration")
    plt.xlabel("Risk")
    plt.ylabel("Duration")
    plt.title("Duration influence on credit risk")
    st.pyplot(fig_box_duration_risk)

    st.subheader("Risk proportion by age and sex")
    data = df.groupby(["Age", "Risk"]).size().unstack().fillna(0)
    st.bar_chart(data)

    st.subheader("Risk proportion by sex")
    data = df.groupby(["Sex", "Risk"]).size().unstack().fillna(0)
    st.bar_chart(data)

