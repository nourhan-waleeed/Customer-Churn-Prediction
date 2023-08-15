# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:43:54 2023

@author: nourhan
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import columns 


model = joblib.load('xgbpipe3.joblib')
st.title('Dinning Churn Prediction')

recency_in_days = st.number_input('How Recent Did this Customer Visit?')
days_since_joining = st.number_input('For How long Has This Cutomer been a Customer')
visit_frequency = st.number_input('Frequency of Visits')
avg_order = st.number_input('Averge Of The Order')
average_time_spent = st.number_input('Average Time Spent In The Restraunt')


def predict(): 
    row = np.array([recency_in_days,days_since_joining,visit_frequency,avg_order,average_time_spent])
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    if prediction[0] == 1: 
        st.success('Will Churn :thumbsdown:')
    else: 
        st.error('Will Not Churn:thumbsup:') 

trigger = st.button('Predict', on_click=predict)