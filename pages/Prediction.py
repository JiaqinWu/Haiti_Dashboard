import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import xgboost as xgb
import plotly.graph_objects as go
import warnings 
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
warnings.filterwarnings('ignore')


# Main page content
st.set_page_config(page_title = 'Haiti EMR System', page_icon='🇭🇹',layout='wide')

# Import Logo
image = "CGHPI.png"

# Function to load data with caching
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load data with caching
df1 = load_data("pages/Datasets/Dataset_Dispense_03_25_2024(1).csv")
df2 = load_data("pages/Datasets/Dataset_HistoricalStatus_03_25_2024(1).csv")
df3 = load_data("pages/Datasets/Dataset_Institution_03_25_2024.csv")
df4 = load_data("pages/Datasets/Dataset_Patientunique_03_25_2024(1).csv")
df5 = load_data("pages/Datasets/Dataset_TestCV_03_25_2024(1).csv")
df6 = load_data("pages/Datasets/Dataset_Visit_03_25_2024(1).csv")
institution = load_data("pages/Datasets/Institution_codebook.csv")

#df1 = load_data("pages/Datasets/Dataset_Dispense_03_25_2024.csv")
#df2 = load_data("pages/Datasets/Dataset_HistoricalStatus_03_25_2024.csv")
#df3 = load_data("pages/Datasets/Dataset_Institution_03_25_2024.csv")
#df4 = load_data("pages/Datasets/Dataset_Patientunique_03_25_2024.csv")
#df5 = load_data("pages/Datasets/Dataset_TestCV_03_25_2024.csv")
#df6 = load_data("pages/Datasets/Dataset_Visit_03_25_2024.csv")
#institution = load_data("pages/Datasets/Institution_codebook.csv")


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
    spreadsheet11 = client.open('Haiti EMR Prediction')
    worksheet11 = spreadsheet11.worksheet('Sheet1')
    sheet1 = pd.DataFrame(worksheet11.get_all_records())
    #st.write(prediction)
except Exception as e:
    st.error(f"Error fetching data from Google Sheets: {str(e)}")


# Use columns for side-by-side layout
col1, col2 = st.columns([1, 6])  # Adjust the width ratio as needed

# Place the image and title in the columns
with col1:
    st.image(image, width=130)

with col2:
    st.title("👩‍⚕️ HIV Treatment Status Prediction")

# Add sidebar
st.sidebar.title('Enter your EMR ID and institution name to match your records')
with st.sidebar:
    emr_id = st.text_input(label='EMR ID:', help="Enter your EMR ID here.")
    inst = st.selectbox(label='Institution Name:', options=institution['INSTITUTION'].tolist(), index=None, placeholder='Select institution...',
                        help="Enter the institution you visit here.")
    st.write("You selected:", inst)
    search_button = st.button("Search",key='search_button')


# Initialize or retrieve session state variables
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {
        'gender': 'Unknown',
        'diag_year': 0.0,
        'facility_same': 'Unknown',
        'year_disp': 0.0,
        'early_prop': 0.0,
        'ontime_prop': 0.0,
        'late_prop': 100.0,
        'dis_late': 0.0,
        'avg_disp': 1.0,
        'avg_nxt_disp': 1.0,
        'yrs_actif': 0.0,
        'num_visit': 0,
        'avg_visit_gap': 1.0,
        'num_hiv': 0,
        'within_one_year': 'Unknown',
        'last_VL_result': 'Unknown'
    }

# Initialize session state for result
if 'result' not in st.session_state:
    st.session_state['result'] = None

# Add conditions
if search_button:
    # Convert 'mpi_ref' to string and create 'joint_id'
    df1['mpi_ref'] = df1['mpi_ref'].astype(str)
    df1['joint_id'] = df1['mpi_ref'] + '_' + df1['EMR_ID'].astype(str)
    df2['mpi_ref'] = [str(i) for i in df2['mpi_ref']]
    df2['joint_id'] = df2['mpi_ref'] + '_' + df2['emr_id']
    df4['mpi_ref'] = df4['mpi_ref'].astype(str)
    df4['joint_id'] = df4['mpi_ref'] + '_' + df4['EMR_ID'].astype(str)
    df6['mpi_ref'] = df6['mpi_ref'].astype(str)
    df6['joint_id'] = df6['mpi_ref'] + '_' + df6['EMR_ID'].astype(str)
    df5['mpi_ref'] = df5['mpi_ref'].astype(str)
    df5['joint_id'] = df5['mpi_ref'] + '_' + df5['EMR_ID'].astype(str)
    df_code = df3[['id_commune', 'commune']].drop_duplicates().rename(columns={'id_commune': 'id_commune_residence', 'commune': 'commune_residence'})
    df40 = pd.merge(df4, df3, left_on='ID_INSTITUTION', right_on='id_institution', how='left')
    df41 = pd.merge(df40, df_code, on='id_commune_residence', how='left').rename(columns={'commune': 'commune_visit'})

    # Filter the joint ID
    joint = df41[(df41['EMR_ID'] == emr_id) & (df41['institution'] == inst)]['joint_id'].values
    #joint = df1[(df1['EMR_ID'] == emr_id) & (df1['INSTITUTION'] == inst)]['joint_id'].values

    # Check if any joint ID is found
    if len(joint) > 0:
        joint = joint[0]

        # Continue processing if joint ID is found
        df41['Sexe'] = df41['Sexe'].replace({'M': 'Male', 'F': 'Female', 'U': 'Unknown'})

        # Merge data frames and process
        df41['date_diagnostic'] = pd.to_datetime(df41['date_diagnostic'])
        df41['same_facility'] = df41.id_commune_residence == df41.id_commune
        df1['dispd'] = pd.to_datetime(df1['dispd'])
        df1['nxt_dispd'] = pd.to_datetime(df1['nxt_dispd'])
        disp_records = df1[df1.joint_id==joint].sort_values(by='dispd')[['dispd','nxt_dispd','statut_rdv','delai_dispense_j']].reset_index(drop=True)
        disp_records['dispd'] = disp_records['dispd'].dt.date
        disp_records['nxt_dispd'] = disp_records['nxt_dispd'].dt.date
        df6['DateVisite'] = pd.to_datetime(df6['DateVisite'])
        visit_records = df6[df6.joint_id == joint].sort_values(by='DateVisite')[['typevisite', 'DateVisite']].reset_index(drop=True)
        visit_records['DateVisite'] = visit_records['DateVisite'].dt.date

        # Check if 'joint_id' exists in df41
        if joint in df41['joint_id'].values:
            gender = df41[df41['joint_id'] == joint]['Sexe'].values[0]
            diag_date = df41[df41['joint_id'] == joint]['date_diagnostic'].values
            if len(diag_date) > 0:
                diag_date = diag_date[0]
                if isinstance(diag_date, np.datetime64):
                    diag_date = diag_date.astype('M8[ms]').astype(datetime)
                    diag_year = round((datetime.now() - diag_date).days / 365.25, 2)
                else:
                    diag_year = 'Unknown'
            else:
                diag_year = 'Unknown'
            commune_visit = df41[df41['joint_id'] == joint]['commune_visit'].values[0]
            commune_residence = df41[df41['joint_id'] == joint]['commune_residence'].values[0]
            # Determine if the facility is the same
            if commune_visit == commune_residence:
                facility_same = 'Yes'
            elif (commune_visit != commune_residence) and pd.notna(commune_visit) and pd.notna(commune_residence):
                facility_same = 'No'
            else:
                facility_same = 'Unknown'
        else:
            gender = 'Unknown'
            diag_year = 0.0
            facility_same = 'Unknown'
        
        min_date = df1[df1.joint_id==joint].dispd.min()
        if disp_records.dispd.count() > 0:
            year_disp = round((datetime.now() - min_date).days / 365.25,2)
        else:
            year_disp = 0.0

        if disp_records.dispd.count() > 1:
            # Proportion of early/on time/late
            total_num = disp_records.dispd.count()-1
            early_num = df1[(df1.joint_id==joint)&(df1.statut_rdv=='Early Refill')].dispd.count()
            early_prop = round(early_num/total_num*100,2)
            ontime_num = df1[(df1.joint_id==joint)&(df1.statut_rdv=='A temps')].dispd.count()
            ontime_prop = round(ontime_num/total_num*100,2)
            late_num = df1[(df1.joint_id==joint)&(df1.statut_rdv.isin(['Après visite ratée','Après IT']))].dispd.count()
            late_prop = round(late_num/total_num*100,2)
            # Number of days usually early ot late
            dis_late = round(df1[df1.joint_id==joint].delai_dispense_j.mean(),2)
            # Average days to next dispensation
            df11 = df1[df1['joint_id'] == joint].sort_values(by='dispd').reset_index()
            df11['dispgap'] = pd.NA
            for i in range(1, df11.shape[0]):
                df11['dispgap'][i] = (df11['dispd'][i] - df11['dispd'][i-1]).days
            avg_disp = round(df11['dispgap'].mean(),2)
            # Average days to next dispensation
            df11['nxt_dispgap'] = pd.NA
            for i in range(1, df11.shape[0]):
                df11['nxt_dispgap'][i] = (df11['nxt_dispd'][i] - df11['dispd'][i-1]).days
            avg_nxt_disp = round(df11['nxt_dispgap'].mean(),2)
        else:
            year_disp = 0.0
            early_prop = 0.0
            ontime_prop = 0.0
            late_prop = 100.0
            dis_late = 0.0
            avg_disp = 1.0
            avg_nxt_disp = 1.0
        
        # Actif years
        df2['date_statut'] = pd.to_datetime(df2['date_statut'])
        status_records = df2[df2['joint_id']==joint][['date_statut','patient_status']].sort_values('date_statut').reset_index().drop(columns='index')
        if status_records.patient_status.count() > 0:
            # Extract the last patient status value as a scalar
            last_patient_status = status_records['patient_status'].iloc[-1]
            if last_patient_status == 'Actif':
                yrs_actif = round((datetime.now() - status_records['date_statut'].iloc[-1]).days / 365.25, 2)
            else:
                yrs_actif = 0.0
        else:
            yrs_actif = 0.0
        
        # Extract the number of visits
        if joint in df6['joint_id'].values:
            num_visit = df6[df6.joint_id == joint].shape[0]
        else:
            num_visit = 0

        # Calculate the average gap between visits
        if num_visit > 1:
            df61 = df6[df6['joint_id'] == joint].sort_values(by='DateVisite').reset_index(drop=True)
            df61['visitgap'] = df61['DateVisite'].diff().dt.days
            avg_visit_gap = round(df61['visitgap'].mean(), 2)
        else:
            avg_visit_gap = 1.0
        
        # Continue processing if joint ID is found
        df5['DateTest'] = pd.to_datetime(df5['DateTest'])
        diagnostic_records = df5[df5.joint_id == joint].sort_values(by='DateTest')[['DateTest', 'Resultat']].reset_index(drop=True)

        num_hiv = diagnostic_records.shape[0]

        latest_test_date = diagnostic_records['DateTest'].max()
        if pd.notna(latest_test_date):
            one_year_ago = datetime.now() - timedelta(days=365)
            within_one_year = 'Yes' if latest_test_date >= one_year_ago else 'No'
        else:
            within_one_year = 'Unknown'

        # Extract recent Viral Load Test result
        last_VL_result = diagnostic_records['Resultat'].iloc[-1] if not diagnostic_records.empty else 'Unknown'

    else:
        gender = 'Unknown'
        diag_year = 0.0
        facility_same = 'Unknown'
        year_disp = 0.0
        early_prop = 0.0
        ontime_prop = 0.0
        late_prop = 100.0
        dis_late = 0.0
        avg_disp = 1.0
        avg_nxt_disp = 1.0
        yrs_actif = 0.0
        num_visit = 0
        avg_visit_gap = 1.0
        num_hiv = 0
        within_one_year = 'Unknown'
        last_VL_result = 'Unknown'

    st.session_state['patient_data'] = {
        'gender': gender,
        'diag_year': diag_year,
        'facility_same': facility_same,
        'year_disp': year_disp,
        'early_prop': early_prop,
        'ontime_prop': ontime_prop,
        'late_prop': late_prop,
        'dis_late': dis_late,
        'avg_disp': avg_disp,
        'avg_nxt_disp': avg_nxt_disp,
        'yrs_actif': yrs_actif,
        'num_visit': num_visit,
        'avg_visit_gap': avg_visit_gap,
        'num_hiv': num_hiv,
        'within_one_year': within_one_year,
        'last_VL_result': last_VL_result
    }

# Retrieve values from session state
patient_data = st.session_state['patient_data']


st.sidebar.markdown("### Personal Information")
sex = st.sidebar.selectbox('Gender', options=['Male', 'Female', 'Unknown'], index=['Male', 'Female', 'Unknown'].index(patient_data['gender']))
diag = st.sidebar.number_input('Years Since Diagnosis', min_value=0.0, step=0.01, format="%.2f", value=patient_data['diag_year'])
facility = st.sidebar.selectbox('Facility Same as Residence', options=['Yes', 'No', 'Unknown'], index=['Yes', 'No', 'Unknown'].index(patient_data['facility_same']))
st.sidebar.markdown("### Dispensation Information")

dispy = st.sidebar.number_input('Years Since First Dispensation', min_value=0.0, step=0.01, format="%.2f", value=patient_data['year_disp'])
dispd = st.sidebar.number_input("Number of days you're usually early or late to get dispensation", step=0.01, format="%.2f", value=patient_data['dis_late'])
prope = st.sidebar.number_input('Proportion of Early Refills (%)', min_value=0.0, max_value=100.0, step=0.01, format="%.2f", value=patient_data['early_prop'])
propo = st.sidebar.number_input('Proportion of On Time Refills (%)', min_value=0.0, max_value=100.0, step=0.01, format="%.2f", value=patient_data['ontime_prop'])
propl = st.sidebar.number_input('Proportion of Late Refills (%)',value=100.0 - prope - propo, min_value=0.0, max_value=100.0, disabled=True)
avgd = st.sidebar.number_input('Average Days Between Dispensations', min_value=0.0, step=0.01, format="%.2f", value=patient_data['avg_disp'])
avgn = st.sidebar.number_input('Average Days Until Next Dispensation', min_value=0.0, step=0.01, format="%.2f", value=patient_data['avg_nxt_disp'])
yeara = st.sidebar.number_input('Years Actively Visiting', min_value=0.0, step=0.01, format="%.2f", value=patient_data['yrs_actif'])

st.sidebar.markdown("### Visit Information")
numberm = st.sidebar.number_input('Average Days Between Visits', min_value=0.0, step=0.01, format="%.2f", value=patient_data['avg_visit_gap'])
numberv = st.sidebar.number_input('Number of Visits', min_value=0, step=1, value=patient_data['num_visit'])

st.sidebar.markdown("### Diagnostics Information")
numbert = st.sidebar.number_input('Number of HIV Visits', min_value=0, step=1, value=patient_data['num_hiv'])
test = st.sidebar.selectbox('Visit Within One Year', options=['Yes', 'No', 'Unknown'], index=['Yes', 'No', 'Unknown'].index(patient_data['within_one_year']))
recent = st.sidebar.selectbox('Last Viral Load Result', options=['Detectable', 'Indetectable', 'Unknown'], index=['Detectable', 'Indetectable', 'Unknown'].index(patient_data['last_VL_result']))

# Define a button to trigger predictions
predict_button = st.sidebar.button("Predict",key='predict_button')
save_button = st.sidebar.button("Save Results", key='save_button') 


# Import the scaler and model
scaler = pickle.load(open("pages/scaler.pkl", "rb"))
model = pickle.load(open('pages/model.pkl', 'rb'))

# Transform the value into 2 lists
lis1 = [diag, dispy, dispd, prope, propo, propl, avgd, avgn, yeara, numbert, numberm, numberv]
fem = int(sex == 'Female')
same = int(facility == 'Yes')
tes = int(test == 'Yes')
res = int(recent == 'Indetectable')
lis2 = [fem, same, tes, res]


# Main content
with st.container():
    st.title("About the App")
    st.write("This app assists in diagnosing the patients' current treatment status from their records. \
             It utilizes the XGBoost model, proven to be the most reliable with over 90% accuracy in our case. \
             The model predicts whether the patient's treatment status is 'Actif' or 'PIT' based on various features \
             such as gender, historical visit date, etc. You can update the measurements by adjusting values \
             using the sliders in the sidebar.")

    # Layout for additional content
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader('Radar Chart for Numeric Variables')
        fig = go.Figure()

        # Add the traces
        fig.add_trace(go.Scatterpolar(
            r=lis1,  
            theta=['Age at Diagnosis', 'Dispensation Years', 'Dispensation Days Early/Late', 'Early Percentage',
                'On-time Percentage', 'Late Percentage', 'Avg Dispensation Gap', 'Avg Days to Next Dispensation',
                'Years in Actif Status', 'Number of HIV Tests', 'Avg Days Between Visits'],
            fill='toself',
            name='Patient'
        ))

        # Update the layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(lis1)] if predict_button else [0,100]  
                )
            ),
            showlegend=True,
            autosize=True
        )

        # Display the radar plot
        st.plotly_chart(fig)
        
    with col2:
        st.subheader('Treatment Status Prediction')

        if predict_button:
            # Combine continuous and categorical variables without scaling 'lis2'
            input_array = np.array(lis1 + lis2).reshape(1, -1)

            # Scale the continuous variables
            input_data_scaled = scaler.transform(input_array[:, :len(lis1)])

            # Combine the scaled continuous variables and categorical variables
            input_data_fin = np.concatenate([input_data_scaled, input_array[:, len(lis1):]], axis=1)

            # Make predictions
            prediction = model.predict(input_data_fin)

            # Display prediction result
            if prediction == 1:
                st.session_state['result'] = 'Actif'
                st.write("<div style='font-size:30px; color:#8B0000;'>Actif</div>", unsafe_allow_html=True)
            else:
                st.session_state['result'] = 'PIT'
                st.write("<div style='font-size:30px; color:#8B0000;'>PIT</div>", unsafe_allow_html=True)

        # Display disclaimer
        st.write("This app assists medical professionals in making a diagnosis, but should not be used as a substitute for professional diagnosis.")

        
if save_button and st.session_state['result'] is not None:
    new_row = {'Date': datetime.now().strftime('%Y-%m-%d'), 'EMR ID': emr_id, 'Institution Name': inst, 'Prediction results': st.session_state['result']}
    new_data = pd.DataFrame([new_row])

    # Append new_data to existing sheet DataFrame
    sheet1 = pd.concat([sheet1, new_data], ignore_index=True)

    try:
        # Clear and update Google Sheets with the updated sheet DataFrame
        #worksheet11.clear()
        worksheet11.update([sheet1.columns.values.tolist()] + sheet1.values.tolist())

        st.sidebar.write("The prediction result has been submitted and Google Sheets updated.")
    except Exception as e:
        st.sidebar.error(f"Error updating Google Sheets: {str(e)}")

