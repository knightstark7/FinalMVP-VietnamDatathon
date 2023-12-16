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
import openai
# set maxuploadsize to 1GB: streamlit run web_copy.py --server.maxUploadSize=1024

#----------Header----------
st.set_page_config(page_title='VietNam Datathon', page_icon='chart_with_upwards_trend',layout='wide')
openai.api_key = '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'

system_msg = "As a helpful assistant who understands data science and data analysis for business. You should present for the business people, don't use any technical terms like the coefficient variation, slope..., your answer should based on a chart, the terms I'm about to give you is just the metrics for you to analyze it, you shouldn't notice any number neither and don't give any reminders or something like that since I want to use your output for my website"

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
    
def classify_and_comments(data, x, y, ranges, comments):
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
        output += '- ' + temp_comments[i][0] + ':\n'
        for j in range(2, len(temp_comments[i])):
            output += '    🔹 ' + temp_comments[i][j] + '\n'
        output += '👉 ' + temp_comments[i][1] + '\n\n'
    with st.expander('See explanation'):
        st.text(output)

def ols(x, y):
    """ 
    I'll use the simple OLS method to find the coefficient 
    of the linear equation: y = beta * x + alpha
    """
    # concat the vector one into x (represent beta)
    x = np.concatenate((np.array(x).reshape(-1, 1), np.ones(x.shape).reshape(-1, 1)), axis=1)
    y = np.array(y).reshape(-1, 1) # convert y vector into a column in 2D matrix
    theta = np.linalg.inv(x.T @ x) @ x.T @ y # theta = ((X^T X) ^ -1) X^T y
    return theta[0][0], theta[1][0] # since the result is a column (1x2)

def time_trend_comments(data, x, y, slope_ranges, cv_ranges, trend_comments, fluctuate_comments):
    # apply linear regression to see the trend of data over time
    beta, _ = ols(data[x], data[y]) # we will ignore the alpha
    comment_type = 0
    is_positive = True
    for i in range(len(slope_ranges) - 1):
        if (i == 0 and abs(beta) <= slope_ranges[0])\
            or (slope_ranges[i] <= beta < slope_ranges[i + 1]):
            comment_type = i
            is_positive = beta > 0
            break
    s = 0 if is_positive else 2
    output = '🔹Trend:' + trend_comments[comment_type][s] + '\n👉 ' + trend_comments[comment_type][s + 1] + '\n'
    
    # calculate Coefficient of Variation - CV to see how the data fluctuate
    cvar = np.std(data[y]) / np.mean(data[y])
    comment_type = 0
    for i in range(len(cv_ranges) - 1):
        if (cv_ranges[i] < cvar <= cv_ranges[i + 1]):
            comment_type = i
            break
    output = '🔹Fluctuation:' + fluctuate_comments[comment_type][0] + '\n👉 ' + fluctuate_comments[comment_type][1] + '\n'
    
    with st.expander('See explanation'):
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
            
            SALES_RANGES = (float('-inf'), 0, .25, .5, 1.1)
            CHANNELS_COMMENTS = [
                ["Total sales are negative, indicating a concerning situation"
                 "Urgent measures are required to address the financial challenges."],
                ["Total sales are within the low range, suggesting a modest performance",
                 "Strategies to boost sales should be reviewed and implemented."],
                [f"Total sales fall within the medium range, indicating a substantial growth",
                "Efforts should be directed towards maintaining and maximizing this positive momentum."],
                [f"Total sales for those channels have good performance",
                "Capitalizing on this success, long-term strategies should focus on sustaining and expanding market share."]
            ]
            
            st.write('Comparing sales in different distribution channels')
            total_sales_per_channel = full_df.groupby('channel_id')['total_sales'].sum().reset_index()
            st.bar_chart(data=total_sales_per_channel, x='channel_id', y='total_sales')
            
            classify_and_comments(total_sales_per_channel, 'channel_id', 'total_sales',
                                  SALES_RANGES, CHANNELS_COMMENTS)

                
            st.write('Comparing sales in different kind of distribution channels')
            total_sales_per_distribution_channel = full_df.groupby('distribution_channel')['total_sales'].sum().reset_index()
            st.bar_chart(data=total_sales_per_distribution_channel, x='distribution_channel', y='total_sales')

            classify_and_comments(total_sales_per_distribution_channel, 'distribution_channel', 'total_sales',
                                  SALES_RANGES, CHANNELS_COMMENTS)
            
            st.write('How do the total sales change over time?')
            
            # format date
            years = full_df['month'].astype(str).str[:4]
            months = full_df['month'].astype(str).str[-2:]
            weeks = full_df['week'].astype(str).str[-2:]

            full_df['year-month-week'] = years.values + '-' + months.values + '-' + weeks.values
            sorted_df = full_df.sort_values(by=['year-month-week'])
            st.line_chart(sorted_df, x='year-month-week', y='total_sales')
            
            SLOPE_RANGES = (.05, .05, .1, float('+inf'))
            TREND_COMMENTS = [
                ["In general, the total sales have a negligible trend over the observed period.",
                 "Consider exploring external influences, diversifying offerings, refining marketing strategies, and focusing on customer retention to potentially stimulate sales growth."],
                
                ["This indicates a moderate positive growth trend in sales over the observed time period.",
                 "Consider expanding into new markets, refining marketing approaches, and strengthening customer relationships to sustain and potentially accelerate sales growth.",
                 "We can see there's a declining trend in total sales.",
                 "Businesses should address some factors, adapt new strategies, and focus on customer retention and market repositioning."],
                
                ["It signals a strong positive growth trend in sales over time.",
                 "Businesses should capitalize on this momentum by optimizing strategies, exploring opportunities, and making strategic investments to sustain that growth.",
                 "We can see there's a strong declining trend in total sales.",
                 "Businesses should implement cost-cutting strategies, reevaluate product offerings, explore innovative approaches."]
            ]
            
            CVAR_RANGES = (0, .3, .6, 1.1)
            # FLUCTUATE_COMMENTS = [
            #     ['The line graph suggests a moderate level of fluctuation in total sales',
            #      'This fluctuation in sales can be overlooked since the nature of the market changes quite a lot over time, so achieving a low to moderate level of fluctuation is a good thing'],
            #     ["This  higher coefficient of variation suggests a more dynamic and variable sales environment"]
            # ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_msg},
                        {"role": "user", "content": f'Coefficient variation between 0.3 and 0.6, give a short observations and suggestions for business about total sales over time'}
            ])
            res = response["choices"][0]["message"]["content"]
            print(res)
            # time_trend_comments()
            
            
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