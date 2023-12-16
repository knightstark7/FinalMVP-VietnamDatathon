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
import copy
import plotly.express as px
from streamlit_lottie import st_lottie

#----------Header----------
st.set_page_config(page_title='VietNam Datathon', page_icon='chart_with_upwards_trend',layout='wide')

with st.container():
    st.title("""Automated Financial Report for Retailers""")
    st.header("Team 16: Datada :wave:", anchor='u')
    st.subheader('Dataset 2: Sales and Inventory Data Of VietNam Retailers',divider='rainbow')
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
        st.subheader("From: VNU HCM - HCMUS")
        st.subheader('Members')
        members = {'Role' : ['Leader','Member','Member','Member','Member'],
                   'Name' : ['Tran Nguyen Huan','Nguyen Duc Tuan Dat','Nguyen Phat Dat','Nguyen Minh Quan','Nguyen Bao Tuan']}
        members_df = pd.DataFrame(members)
        members_df.index = members_df.index + 1
        st.table(members_df)
    with right_columns:
        st_lottie(data_lottie, height=400, key='coding')
    
def classify_and_comments(ranges, comments, data, x, y):
    maxvals = data[y].max()
    temp_comments = copy.deepcopy(comments)
    for index, row in data.iterrows():
        for i in range(len(ranges) - 1):
            perc = row[y] / maxvals
            if (ranges[i] <= perc < ranges[i + 1]):
                temp_comments[i].append(row[x])
                break
    output = ""
    for i in range(len(temp_comments)):
        if len(temp_comments[i]) <= 2:
            continue
        output += '- ' + temp_comments[i][0]
        for j in range(2, len(temp_comments[i])):
            output += '    🔹 ' + temp_comments[i][j] + '\n'
        output += '👉 ' + temp_comments[i][1] + '\n\n'
    with st.expander('See explanationss'):
        st.text(output)

# ----------sales----------
with st.container(border=True):
    st.header('Sales', divider='rainbow')

    uploaded_files = st.file_uploader("Upload your sales data (*.xlsx) here", 
                                    accept_multiple_files=True)
    if uploaded_files:
        full_df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            dataframe = pd.read_excel(uploaded_file)
            full_df = pd.concat([full_df, dataframe])
        
        full_df['total_sales'] = full_df['sold_quantity'] * full_df['net_price']
         
        st.write("Here's your merged sales data")
        st.write(full_df)
    
        if st.button('Press here to analyze your sales data'):
            st.subheader("Some insights about your data")
            
            SALES_RANGES = (float('-inf'), 0, .25, .50, 1.1)
            CHANNELS_COMMENTS = [
                ["Total sales are negative, indicating a concerning situation:\n"
                 "Urgent measures are required to address the financial challenges."],
                ["Total sales are within the low range, suggesting a modest performance:\n",
                 "Strategies to boost sales should be reviewed and implemented."],
                [f"Total sales fall within the medium range, indicating a substantial growth:\n",
                "Efforts should be directed towards maintaining and maximizing this positive momentum."],
                [f"Total sales for those channels have good performance:\n",
                "Capitalizing on this success, long-term strategies should focus on sustaining and expanding market share."]
            ]
            
            st.write('Comparing sales in different distribution channels')
            total_sales_per_channel = full_df.groupby('channel_id')['total_sales'].sum().reset_index()
            st.bar_chart(data=total_sales_per_channel, x='channel_id', y='total_sales')
            
            classify_and_comments(SALES_RANGES, CHANNELS_COMMENTS, 
                                  total_sales_per_channel, 'channel_id', 'total_sales')

                
            st.write('Comparing sales in different kind of distribution channels')
            total_sales_per_distribution_channel = full_df.groupby('distribution_channel')['total_sales'].sum().reset_index()
            st.bar_chart(data=total_sales_per_distribution_channel, x='distribution_channel', y='total_sales')

            classify_and_comments(SALES_RANGES, CHANNELS_COMMENTS, 
                                  total_sales_per_distribution_channel, 'distribution_channel', 'total_sales')
            
            
            
            st.subheader("Sales forecast")

            st.subheader("Some suggested solutions for you")
        
        
    else:
        pass


# ----------inventory----------
with st.container(border=True):    
    st.header('Inventory', divider='rainbow')

    uploaded_files = st.file_uploader("Upload your inventory data (*.xlsx) here", 
                                    accept_multiple_files=True)

    if uploaded_files:
        full_df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_excel(uploaded_file)
            
            full_df = pd.concat([full_df, dataframe])
            
        st.write("Here's your merged inventory data")
        st.write(full_df)
        
        if st.button('Press here to analyze your inventory data'): 
            st.subheader("Some insights about your data")

            st.subheader("Current inventory report and some feasible solutions")
        else:
            pass

# ----------Product----------
with st.container(border=True):    
    st.header('Products', divider='rainbow')

    uploaded_files = st.file_uploader("Upload your products data (*.xlsx) here", 
                                    accept_multiple_files=True)

    if uploaded_files:
        full_df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            
            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_excel(uploaded_file)
            
            full_df = pd.concat([full_df, dataframe])
            
        st.write("Here's your merged products data")
        st.write(full_df)
        
        if st.button('Press here to analyze your products data'):
            st.subheader("Some insights about your data")
            st.subheader("")
        else:
            pass