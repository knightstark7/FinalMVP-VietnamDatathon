import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# from keras.models import load_model
import os
import glob
import re

st.set_page_config(
    page_title="Final MVP Datada",
    page_icon=":report:",
    layout="wide",
)

st.title("""Automated Financial Report for Retailers""")

# initialize models


models_files = glob.glob("../models/lstm_sales_*.keras")
print(models_files)

channels = [re.findall(r'lstm_sales_(\w+).keras', filename) for filename in models_files]
# print(channels)

# ltsm_models = load_model('../models/')


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
        
    st.subheader("Some insights about your data")

    st.subheader("Sales forecast")

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

