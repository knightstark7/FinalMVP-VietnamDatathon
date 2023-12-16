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
    

# left, right = st.columns(2)
        
# # Generate data for the first scatter plot
# xs1 = [random.randint(0, 10) for _ in range(100)]
# ys1 = [random.randint(0, 10) for _ in range(100)]

# # Create the first scatter plot
# with left:
#     fig, ax = plt.subplots()
#     ax.scatter(xs1, ys1)
#     ax.set_xlabel("X")  
#     ax.set_ylabel("Y")  
#     ax.set_title("Random plot1") 
#     st.pyplot(fig)

# # Generate data for the second scatter plot
# xs2 = [random.randint(0, 10) for _ in range(100)]
# ys2 = [random.randint(0, 10) for _ in range(100)]

# # Create the second scatter plot
# with right:
#     fig, ax = plt.subplots()
#     ax.scatter(xs2, ys2)
#     ax.set_xlabel("X")  
#     ax.set_ylabel("Y")  
#     ax.set_title("Random plot2")  
#     st.pyplot(fig)

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
            
            st.write('Comparing sales in different distribution channels')
            total_sales_per_channel = full_df.groupby('channel_id')['total_sales'].sum().reset_index()
            st.bar_chart(data=total_sales_per_channel, x='channel_id', y='total_sales')

            with st.expander('See explanation'):
                SALES_RANGES = (float('-inf'), 0, 1e10, 3e10, float('inf'))
                type_comments = [["Total sales are negative, indicating a concerning situation:\n"
                                "Urgent measures are required to address the financial challenges."],
                                ["Total sales are within the low range, suggesting a modest performance:\n",
                                "Strategies to boost sales should be reviewed and implemented."],
                                [f"Total sales fall within the medium range, indicating a substantial growth:\n",
                                "Efforts should be directed towards maintaining and maximizing this positive momentum."],
                                [f"Total sales for those channels have an exceptional performance:\n",
                                "Capitalizing on this success, long-term strategies should focus on sustaining and expanding market share."]]
                
                for index, row in total_sales_per_channel.iterrows():
                    for i in range(len(SALES_RANGES) - 1):
                        if (SALES_RANGES[i] <= row['total_sales'] < SALES_RANGES[i + 1]):
                            type_comments[i].append(row['channel_id'])
                            break
                output = ""
                for i in range(len(type_comments)):
                    if len(type_comments[i]) <= 2:
                        continue
                    output += type_comments[i][0]
                    for j in range(2, len(type_comments[i])):
                        output += '    * ' + type_comments[i][j] + '\n'
                    output += '=> ' + type_comments[i][1] + '\n'
                st.text(output)

            st.write('Comparing sales in different kind of distribution channels')
            total_sales_per_distribution_channel = full_df.groupby('distribution_channel')['total_sales'].sum().reset_index()
            st.bar_chart(data=total_sales_per_distribution_channel, x='distribution_channel', y='total_sales')




            st.subheader("Sales forecast")

            st.subheader("Some suggested solutions for you")
        
        
    else:
        pass


# ----------inventory----------
with st.container(border=True):    
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

    if uploaded_files is not None:
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