import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Final MVP Datada",
    page_icon="✅",
    layout="wide",
)

st.title("""Automated Financial Report for Retailers""")

# sales
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

