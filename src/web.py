import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from keras.models import load_model
import os
import glob
import re

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from keras.models import Sequential
from keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Final MVP Datada",
    page_icon=":report:",
    layout="wide",
)

st.title("""Automated Financial Report for Retailers""")

# initialize models
models_files = glob.glob("../models/lstm_sales_*.keras")
print(models_files)

channels = [re.findall(r'lstm_sales_(\w+).keras', filename)[0] for filename in models_files]
print(channels)


# sales
with st.container():
    st.header('Sales', divider='rainbow')

    uploaded_files = st.file_uploader("Upload your sales data (*.xlsx) here", 
                                    accept_multiple_files=True)

    if uploaded_files is not None:
        full_df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_excel(uploaded_file)
            
            full_df = pd.concat([full_df, dataframe])
            
        st.write("Here's your merged sales data")
        st.write(full_df)
        
    full_df['total_cost'] = full_df['sold_quantity'] * full_df['cost_price']
    full_df['total_sales'] = full_df['sold_quantity'] * full_df['net_price']
    full_df['profit'] = (full_df['sold_quantity'] >= 0).mul(full_df['total_sales']) - full_df['total_cost']
    
    st.subheader("Some insights about your data")

    st.subheader("Sales forecast")
    
    # first channel
    ltsm_models = load_model(models_files[0])
    # ltsm_models.compile(optimizer='adam', loss='mse')
    
    channel1 = full_df[full_df['channel_id'] == channels[0]]['profit']
    
    train_values = np.array(channel1).reshape(-1, 1, 1)
    test_values = np.array(test['profit']).reshape(-1, 1, 1)
    
    ltsm_models.fit(train_values, channel1, epochs=200, verbose=0)
    
    pred = ltsm_models.predict(channel1[:, channel1.columns != 'profit'])
    
    fig, axs = plt.subplots(2, 2, figsize=(25, 12))
    
    axs[0, 0].plot(test.index, forecast_values_test, label='LSTM Forecast on Test Set')
    axs[0, 0].set_title(f'LSTM Forecast of Future Profit Over Time for Channel {channel}')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Profit')
    axs[0, 0].legend()
    
    st.line_chart()
    st.subheader("Some suggested solutions for you")

# inventory
with st.container():
    st.header('Inventory', divider='rainbow')

    uploaded_files = st.file_uploader("Upload your inventory data (*.xlsx) here", 
                                    accept_multiple_files=True)

    if uploaded_files is not None:
        full_df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_excel(uploaded_file)
            
            full_df = pd.concat([full_df, dataframe])
            
        st.write("Here's your merged inventory data")
        st.write(full_df)
        
    st.subheader("Some insights about your data")

    st.subheader("Current inventory report and some feasible solutions")

