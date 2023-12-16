import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# from keras.models import load_model
import os
import glob
import re
import random
import requests
from streamlit_lottie import st_lottie

#----------Header----------
st.set_page_config(page_title='VietNam Datathon', page_icon='chart_with_upwards_trend',layout='wide')

with st.container():
    st.header("Group 16: Datada :wave:",divider='rainbow')
    st.title('Dataset 2: Sales and Inventory Data Of VietNam Retailers')
    st.write('Goals:...')

# ----------LOAD ASSETS----------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


#----------Information----------
with st.container():
    st.header("Information :pencil:",divider='rainbow')
    # st.subheader("Information of team :wave:")
    left_columns, right_columns = st.columns(2)
    
    data_lottie = load_lottie('https://lottie.host/b8d0bada-7671-4c40-8cce-9cdeb60820b0/FICpxvKiTx.json')
    with left_columns:
        st.header('Members')
        members = {'Role' : ['Leader','Member','Member','Member','Member'],
                   'Name' : ['Tran Nguyen Huan','Nguyen Duc Tuan Dat','Nguyen Phat Dat','Nguyen Minh Quan','Nguyen Bao Tuan'],
                   'School': ['HCMUS','HCMUS','HCMUS','HCMUS','HCMUS']}
        members_df = pd.DataFrame(members)
        st.table(members_df)
    with right_columns:
        st_lottie(data_lottie, height=400, key='coding')
    
# ----------sales----------
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
    
    left, right = st.columns(2)
    
    # Generate data for the first scatter plot
    xs1 = [random.randint(0, 10) for _ in range(100)]
    ys1 = [random.randint(0, 10) for _ in range(100)]

    # Create the first scatter plot
    with left:
        fig, ax = plt.subplots()
        ax.scatter(xs1, ys1)
        ax.set_xlabel("X")  
        ax.set_ylabel("Y")  
        ax.set_title("Random plot1") 
        st.pyplot(fig)

    # Generate data for the second scatter plot
    xs2 = [random.randint(0, 10) for _ in range(100)]
    ys2 = [random.randint(0, 10) for _ in range(100)]

    # Create the second scatter plot
    with right:
        fig, ax = plt.subplots()
        ax.scatter(xs2, ys2)
        ax.set_xlabel("X")  
        ax.set_ylabel("Y")  
        ax.set_title("Random plot2")  
        st.pyplot(fig)


# ----------inventory----------
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

