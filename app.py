# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 02:45:35 2024

@author: HabeebullahiLawal
"""

import streamlit as st
from predict_page import show_predict_page
from explore import explore_data



page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    explore_data()
