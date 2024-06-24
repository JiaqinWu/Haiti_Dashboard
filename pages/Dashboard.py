import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from streamlit_gsheets import GSheetsConnection
from datetime import datetime, timedelta
from millify import millify # shortens values (10_000 ---> 10k)
from streamlit_extras.metric_cards import style_metric_cards # beautify metric card with css
import plotly.graph_objects as go
import altair as alt 
#import seaborn as sns
#import plotnine
#from plotnine import *
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import warnings 
warnings.filterwarnings('ignore')

image = "CGHPI.png"
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
#creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
# Use Streamlit's secrets management
creds_dict = st.secrets["gcp_service_account"]
#creds_json = json.dumps(creds_dict)
#creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
#client = gspread.authorize(creds)
#prediction = pd.DataFrame(client.open('Haiti EMR Prediction').worksheet('Sheet1').get_all_records())

# Extract individual attributes needed for ServiceAccountCredentials
credentials = {
    "type": creds_dict.type,
    "project_id": creds_dict.project_id,
    "private_key_id": creds_dict.private_key_id,
    "private_key": creds_dict.private_key,
    "client_email": creds_dict.client_email,
    "client_id": creds_dict.client_id,
    "auth_uri": creds_dict.auth_uri,
    "token_uri": creds_dict.token_uri,
    "auth_provider_x509_cert_url": creds_dict.auth_provider_x509_cert_url,
    "client_x509_cert_url": creds_dict.client_x509_cert_url,
}

# Create JSON string for credentials
creds_json = json.dumps(credentials)

# Load credentials and authorize gspread
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
client = gspread.authorize(creds)

# Example usage: Fetch data from Google Sheets
try:
    spreadsheet = client.open('Haiti EMR Prediction')
    worksheet = spreadsheet.worksheet('Sheet1')
    prediction = pd.DataFrame(worksheet.get_all_records())
    st.write(prediction)
except Exception as e:
    st.error(f"Error fetching data from Google Sheets: {str(e)}")

#prediction = pd.read_csv("pages/Datasets/EMR_prediction.csv")


# Main page content
st.set_page_config(page_title = 'Haiti Prediction Dashboard', page_icon='🧑‍⚕️',layout='wide')

# Use columns for side-by-side layout
col1, col2 = st.columns([1, 6])  # Adjust the width ratio as needed

# Place the image and title in the columns
with col1:
    st.image(image, width=130)

with col2:
    st.title('🧑‍⚕️ Haiti Prediction Dashboard')

"""
Please use this tab to track the latest treatment status prediction results!
"""

# Prediction tab
# creates the container for metric card
dash_1 = st.container()

with dash_1:
    # Get Description data
    total_ppl = prediction['EMR ID'].count()
    total_a = prediction[prediction['Prediction results']=='Actif']['EMR ID'].count()
    total_p = prediction[prediction['Prediction results']=='PIT']['EMR ID'].count()

    col1, col2, col3 = st.columns(3)
    # create column span
    col1.metric(label="Total Number", value= millify(total_ppl, precision=2))

    col2.metric(label="Actif Number", value= millify(total_a, precision=2))

    col3.metric(label="PIT Number", value= millify(total_p, precision=2))

    # this is used to style the metric card
    style_metric_cards(border_left_color="#DBF227")

# creates the container for metric card
dash_2 = st.container()

with dash_2:
    # Get Description data
    prediction['Date'] = pd.to_datetime(prediction['Date'])
    # Extract the date part only
    prediction['Date'] = prediction['Date'].dt.date
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)

    today_ppl = prediction[prediction['Date'] == today]['EMR ID'].count()
    yesterday_ppl = prediction[prediction['Date'] == yesterday]['EMR ID'].count()
    lastweek_ppl = prediction[prediction['Date'] == last_week]['EMR ID'].count()


    col1, col2, col3 = st.columns(3)
    # create column span
    col1.metric(label="Today's Number", value= millify(today_ppl, precision=2))

    col2.metric(label="Yesterday's Number", value= millify(yesterday_ppl, precision=2))

    col3.metric(label="Last Week's Number", value= millify(lastweek_ppl, precision=2))

    # this is used to style the metric card
    style_metric_cards(border_left_color="#DBF227")

# creates the container
dash_3 = st.container()

with dash_3:
    col1, col2 = st.columns(2)
    with col1:
        date_values = prediction.groupby('Date')['EMR ID'].count().reset_index().rename(columns={'EMR ID':'Number'})
        st.markdown("#### The Daily Prediction Number:")
        st.dataframe(date_values)

    with col2:
        ins_values = prediction.groupby('Institution name')['EMR ID'].count().reset_index().rename(columns={'EMR ID':'Number'})
        st.markdown("#### The Prediction Number by Institution:")
        st.dataframe(ins_values)
# creates the container
dash_4 = st.container()

with dash_4:
    col1, col2 = st.columns(2)
    with col1:
        dateins_values = prediction.groupby(['Date','Prediction results'])['EMR ID'].count().reset_index().rename(columns={'EMR ID':'Number'})
        st.markdown("#### The Daily Prediction Number by Institution:")
        st.dataframe(dateins_values)

    with col2:
        datestat_values = prediction.groupby(['Institution name','Prediction results'])['EMR ID'].count().reset_index().rename(columns={'EMR ID':'Number'})
        st.markdown("#### The Prediction Number by Institution and Treatment Status:")
        st.dataframe(datestat_values)

# creates the container
dash_5 = st.container()

with dash_5:
    col1, col2 = st.columns(2)
    top_instution = prediction.groupby('Institution name')['EMR ID'].count().reset_index().rename(columns={'Institution name':'Institution',\
                                                                                                            'EMR ID':'Number'})
    top_instution = top_instution.nlargest(10, 'Number')

    prediction1 = prediction[prediction['Prediction results']== 'PIT']
    top_PIT = prediction1.groupby('Institution name')['EMR ID'].count().reset_index().rename(columns={'Institution name':'Institution',\
                                                                                                            'EMR ID':'Number'})
    top_PIT = top_PIT.nlargest(10, 'Number')

    # create the altair chart for top occupations
    with col1:
        chart = alt.Chart(top_instution).mark_bar(opacity=0.9,color="#9FC131").encode(
                x=alt.X('sum(Number):Q', title='Number of Patients'),  # Rename x-axis
                y=alt.Y('Institution:N', sort='-x', title='Institution Name')  # Rename y-axis 
            )
        chart = chart.properties(title="Number of Patients in Top 10 Institution" )

        st.altair_chart(chart,use_container_width=True)
    
    with col2:
        chart = alt.Chart(top_PIT).mark_bar(opacity=0.9,color="#9FC131").encode(
                x=alt.X('sum(Number):Q', title='Number of Patients'),  # Rename x-axis
                y=alt.Y('Institution:N', sort='-x', title='Institution Name')  # Rename y-axis 
            )
        chart = chart.properties(title="Number of PIT Patients in Top 10 Institution" )

        st.altair_chart(chart,use_container_width=True)

    
