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


# Main page content
st.set_page_config(page_title = 'Haiti EMR System', page_icon='🇭🇹',layout='wide')

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

# Load data
#df1 = pd.read_csv("pages/Datasets/Dataset_Dispense_03_25_2024.csv")
#df2 = pd.read_csv("pages/Datasets/Dataset_HistoricalStatus_03_25_2024.csv")
#df3 = pd.read_csv("pages/Datasets/Dataset_Institution_03_25_2024.csv")
#df4 = pd.read_csv("pages/Datasets/Dataset_Patientunique_03_25_2024.csv")
#df5 = pd.read_csv("pages/Datasets/Dataset_TestCV_03_25_2024.csv")
#df6 = pd.read_csv("pages/Datasets/Dataset_Visit_03_25_2024.csv")
#scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
#creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)

# Use Streamlit's secrets management
#creds_dict = st.secrets["gcp_service_account"]
#creds_json = json.dumps(creds_dict)
#creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
#client = gspread.authorize(creds)
 
#df4 = pd.DataFrame(client.open('Dataset_Patient_Unique_06-28-23').get_worksheet(0).get_all_records())
#institution = pd.read_csv("pages/Datasets/Institution_codebook.csv")
image = "CGHPI.png"
sheet = pd.DataFrame(columns=['Date', 'EMR ID', 'Institution Name', 'Comments'])

#ssl._create_default_https_context = ssl._create_stdlib_context
# Create a connection object.
#conn = st.connection("gsheets", type=GSheetsConnection)
#sheet = conn.read(spreadsheet = "https://docs.google.com/spreadsheets/d/1RgNkeB9F0PyOZsjNRc6VCRgHccRAnwQ1rWVJXC2lYXI/edit?gid=0#gid=0")

# Use columns for side-by-side layout
col1, col2 = st.columns([1, 6])  # Adjust the width ratio as needed

# Place the image and title in the columns
with col1:
    st.image(image, width=130)

with col2:
    st.title('🇭🇹 Haiti EMR System')
    st.sidebar.title('Enter your EMR ID and institution name to match your records')

"""
We're very excited to release Predictive Model used for Haiti HIV patients!
"""


# Create an empty DataFrame for comments
if "sheet" not in st.session_state:
    st.session_state.sheet = pd.DataFrame(columns=['Date', 'EMR ID', 'Institution Name', 'Comments'])

if "search_results" not in st.session_state:
    st.session_state.search_results = None



# Sidebar inputs
with st.sidebar:
    emr_id = st.text_input(label='EMR ID:', help="Enter your EMR ID here.")
    inst = st.selectbox(label='Institution Name:', options=institution['INSTITUTION'].tolist(), index=None, placeholder='Select institution...',
                        help="Enter the institution you visit here.")
    st.write("You selected:", inst)
    search_button = st.button("Search")
    comment = st.text_area("Comments:", key="comment", help="Enter your comments here.")
    submit_comment = st.button("Submit Comment", key="submit_comment")

    if submit_comment:
        if comment:
            # Add comment to DataFrame
            new_row = {'Date': datetime.now().strftime('%Y-%m-%d'), 'EMR ID': emr_id, 'Institution Name': inst, 'Comments': comment}
            st.session_state.sheet = st.session_state.sheet.append(new_row, ignore_index=True)
            sheet = st.session_state.sheet
            st.write("Your comment has been submitted.")
        else:
            st.write("Please enter a comment before submitting.")

    # Download the comments as a CSV file
    st.download_button(
        label="Download Comments as CSV",
        data=st.session_state.sheet.to_csv(index=False).encode('utf-8'),
        file_name='comments.csv',
        mime='text/csv'
    )


# Create five new tabs with centered labels
listTabs = ['😷 Patient', '💊 Dispensation', '🏥 Visit', '🥼 Diagnostics' ]
tab1, tab2, tab3, tab4 = st.tabs([f'{s}&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;' for s in listTabs])

# Function to perform search and save results to session state
def perform_search(emr_id, inst):
    df1['mpi_ref'] = df1['mpi_ref'].astype(str)
    df1['joint_id'] = df1['mpi_ref'] + '_' + df1['EMR_ID'].astype(str)
    joint = df1[(df1['EMR_ID'] == emr_id) & (df1['INSTITUTION'] == inst)]['joint_id'].values

    if len(joint) > 0:
        joint_id = joint[0]
        # Additional processing and merge logic as needed, assuming results in df1, df4, df5, df6, etc.
        return joint_id
    else:
        return None

# Search action
if search_button:
    joint_id = perform_search(emr_id, inst)
    st.session_state.search_results = joint_id


# Patient tab
with tab1:
    st.markdown("### 😷 Welcome to the Patient tab!")
    st.markdown("#### This tab will provide patient-level information, including gender, birthdate, diagnosis date, and details of the facility visited and place of residence.")

    if st.session_state.search_results:
        joint = st.session_state.search_results

        # Continue processing if joint ID is found
        df1['mpi_ref'] = df1['mpi_ref'].astype(str)
        df1['joint_id'] = df1['mpi_ref'] + '_' + df1['EMR_ID'].astype(str)
        df4['mpi_ref'] = df4['mpi_ref'].astype(str)
        df4['joint_id'] = df4['mpi_ref'] + '_' + df4['EMR_ID'].astype(str)
        df4['Sexe'] = df4['Sexe'].replace('M', 'Male').replace('F', 'Female').replace('U', 'Unknown')

        # Merge data frames and process
        df_code = df3[['id_commune', 'commune']].drop_duplicates().rename(columns={'id_commune': 'id_commune_residence', 'commune': 'commune_residence'})
        df40 = pd.merge(df4, df3, left_on='ID_INSTITUTION', right_on='id_institution', how='left')
        df41 = pd.merge(df40, df_code, on='id_commune_residence', how='left').rename(columns={'commune': 'commune_visit'})
        df41['date_diagnostic'] = pd.to_datetime(df41['date_diagnostic'])
        df41['same_facility'] = df41.id_commune_residence == df41.id_commune

        # Container for displaying data
        dash_1 = st.container()

        with dash_1:
            col1, col2 = st.columns(2)
            with col1:
                # Extract gender
                gender = df41[df41['joint_id'] == joint]['Sexe'].values[0]

                # Check if gender is found and display
                if len(gender) > 0:
                    st.markdown(f"##### Gender: {gender}")
                else:
                    st.markdown("##### Gender information not found.")

                # Extract birthday
                datenaissance = df41[df41['joint_id'] == joint]['Datenaissance'].values[0]

                # Check if birthday is found and display
                if pd.isna(datenaissance) == False:
                    st.markdown(f"##### Birthday: {datenaissance}")
                else:
                    st.markdown("##### Birthday information not found.")

                # Extract Diagnostic Date
                diag_date = df41[df41['joint_id'] == joint]['date_diagnostic'].values

                # Check if Diagnostic Date is found and display
                if len(diag_date) > 0:
                    diag_date = diag_date[0]
                    if isinstance(diag_date, np.datetime64):
                        diag_date = diag_date.astype('M8[ms]').astype(datetime)
                    diag_date_str = diag_date.strftime('%Y-%m-%d')
                    diag_year = round((datetime.now() - diag_date).days / 365.25, 2)
                    st.markdown(f"##### First Diagnosis Date: {diag_date_str}")
                    st.markdown(f"##### Years since Diagnosis: {diag_year}")
                else:
                    st.markdown("##### Diagnostic Date not found.")

            with col2:
                # Extract commune_visit information
                commune_visit = df41[df41['joint_id'] == joint]['commune_visit'].values[0]

                # Check if commune_visit is found and display
                if pd.isna(commune_visit) == False:
                    st.markdown(f"##### Commune of Visit: {commune_visit}")
                else:
                    st.markdown("##### Commune of visit information not found.")

                # Extract commune_residence information
                commune_residence = df41[df41['joint_id'] == joint]['commune_residence'].values[0]

                # Check if commune_residence is found and display
                if pd.isna(commune_residence) == False:
                    st.markdown(f"##### Commune of Residence: {commune_residence}")
                else:
                    st.markdown("##### Commune of residence information not found.")
                    
    else:
        st.markdown("#### No matching records found for the given EMR ID and institution.")

# Dispensation tab
with tab2:
    st.markdown("### 💊 Welcome to the Dispensation tab!")
    st.markdown("#### This tab will provide dispensation-level information, including historical dispensation records, the first dispensation date, the number of years receiving dispensation, the percentage of early/on-time/late dispensations, average early or late dispensation days, average dispensation gap, historical treatment records, and the number of Actif years.")

    
    if st.session_state.search_results:
        joint = st.session_state.search_results

        df2['mpi_ref'] = [str(i) for i in df2['mpi_ref']]
        df2['joint_id'] = df2['mpi_ref'] + '_' + df2['emr_id']
        df1['dispd'] = pd.to_datetime(df1['dispd'])
        df1['nxt_dispd'] = pd.to_datetime(df1['nxt_dispd'])
        disp_records = df1[df1.joint_id==joint].sort_values(by='dispd')[['dispd','nxt_dispd','statut_rdv','delai_dispense_j']].reset_index(drop=True)
        disp_records['dispd'] = disp_records['dispd'].dt.date
        disp_records['nxt_dispd'] = disp_records['nxt_dispd'].dt.date

        # Container for displaying data
        dash_1 = st.container()

        with dash_1:
            col1, col2 = st.columns(2)
            with col1:
                # Display the dispensation records in a pretty table
                st.markdown("##### Your Dispensation Records are shown below:")
                st.dataframe(disp_records)            
        
                # Number of years been getting dispensation
                min_date = df1[df1.joint_id==joint].dispd.min()
                if disp_records.dispd.count() > 0:
                    year_disp = round((datetime.now() - min_date).days / 365.25,2)
                    min_date_str = min_date.strftime('%Y-%m-%d')
                    st.markdown(f"##### The 1st time you got dispensation is {min_date_str}")
                    st.markdown(f"##### Number of years you've been getting dispensation is {year_disp} years")
                else:
                    st.markdown("#### Dispensation records not found")
                
                if disp_records.dispd.count() > 1:
                    # Proportion of early/on time/late
                    total_num = disp_records.dispd.count()-1
                    early_num = df1[(df1.joint_id==joint)&(df1.statut_rdv=='Early Refill')].dispd.count()
                    early_prop = round(early_num/total_num*100,2)
                    ontime_num = df1[(df1.joint_id==joint)&(df1.statut_rdv=='A temps')].dispd.count()
                    ontime_prop = round(ontime_num/total_num*100,2)
                    late_num = df1[(df1.joint_id==joint)&(df1.statut_rdv.isin(['Après visite ratée','Après IT']))].dispd.count()
                    late_prop = round(late_num/total_num*100,2)
                    st.markdown(f"##### Percentage of times you're early to get dispensation is {early_prop}%")
                    st.markdown(f"##### Percentage of times you're on-time to get dispensation is {ontime_prop}%")
                    st.markdown(f"##### Percentage of times you're late to get dispensation is {late_prop}%")
                    # Number of days usually early ot late
                    dis_late = round(df1[df1.joint_id==joint].delai_dispense_j.mean(),2)
                    st.markdown(f"##### Number of days you're usually early or late to get dispensation is {dis_late} days")
                else:
                    st.markdown("#### Dispensation records not enough")
            
            with col2:
        
                if disp_records.dispd.count() > 1:
                    # Average days to next dispensation
                    df11 = df1[df1['joint_id'] == joint].sort_values(by='dispd').reset_index()
                    df11['dispgap'] = pd.NA
                    for i in range(1, df11.shape[0]):
                        df11['dispgap'][i] = (df11['dispd'][i] - df11['dispd'][i-1]).days
                    avg_disp = round(df11['dispgap'].mean(),2)
                    st.markdown(f"##### Average days between your recent two dispensations is {avg_disp} days")
                    # Average days to next dispensation
                    df11['nxt_dispgap'] = pd.NA
                    for i in range(1, df11.shape[0]):
                        df11['nxt_dispgap'][i] = (df11['nxt_dispd'][i] - df11['dispd'][i-1]).days
                    avg_nxt_disp = round(df11['nxt_dispgap'].mean(),2)
                    st.markdown(f"##### Average days to your next dispensation is {avg_nxt_disp} days")
                else:
                    st.markdown("##### Not enough visit records to calculate the average dispensation gap.")

                # Actif years
                df2['date_statut'] = pd.to_datetime(df2['date_statut'])
                status_records = df2[df2['joint_id']==joint][['date_statut','patient_status']].sort_values('date_statut').reset_index().drop(columns='index')
                status_records['date_statut'] = status_records['date_statut'].dt.date
                st.markdown("##### Your Treatment Status Records are shown below:")
                st.dataframe(status_records)

                if status_records.patient_status.count() > 0:
                    # Extract the last patient status value as a scalar
                    last_patient_status = status_records['patient_status'].iloc[-1]

                    if last_patient_status == 'Actif':
                        yrs_actif = round((datetime.now() - status_records['date_statut'].iloc[-1]).days / 365.25, 2)
                    else:
                        yrs_actif = 0

                    st.markdown(f"##### Number of years you've been in Actif status is {yrs_actif} years")
                else:
                    st.markdown("##### Treatment status historical records not found.")

    else:
        st.markdown("#### No matching records found for the given EMR ID and institution.")

        



# Visit tab
with tab3:
    st.markdown("### 🏥 Welcome to the Visit tab!")
    st.markdown("#### This tab will provide visit-level information, including historical visit records, number of visits and average gap between visits.")

    if st.session_state.search_results:
        joint = st.session_state.search_results

        df6['mpi_ref'] = df6['mpi_ref'].astype(str)
        df6['joint_id'] = df6['mpi_ref'] + '_' + df6['EMR_ID'].astype(str)
        # Continue processing if joint ID is found
        df6['DateVisite'] = pd.to_datetime(df6['DateVisite'])
        visit_records = df6[df6.joint_id == joint].sort_values(by='DateVisite')[['typevisite', 'DateVisite']].reset_index(drop=True)
        visit_records['DateVisite'] = visit_records['DateVisite'].dt.date

        # Container for displaying data
        dash_1 = st.container()

        with dash_1:
            col1, col2 = st.columns(2)
            with col1:

                # Display the visit records in a pretty table
                st.markdown("##### Your Visit Records are shown below:")
                st.dataframe(visit_records)
            

            with col2:
                # Extract the number of visits
                if joint in df6['joint_id'].values:
                    num_visit = df6[df6.joint_id == joint].shape[0]
                else:
                    num_visit = 0

                # Check if the number of visits is found and display
                if num_visit > 0:
                    st.markdown(f"##### Number of times you've visited: {num_visit}")
                else:
                    st.markdown("##### Visit historical records not found.")

                # Calculate the average gap between visits
                if num_visit > 1:
                    df61 = df6[df6['joint_id'] == joint].sort_values(by='DateVisite').reset_index(drop=True)
                    df61['visitgap'] = df61['DateVisite'].diff().dt.days
                    avg_visit_gap = round(df61['visitgap'].mean(), 2)

                    st.markdown(f"##### Average days between your visits: {avg_visit_gap} days")
                else:
                    st.markdown("##### Not enough visit records to calculate the average visit gap.")
    else:
        st.markdown("#### No matching records found for the given EMR ID and institution.")



# Diagnostics tab
with tab4:
    st.markdown("### 🥼Welcome to the Diagnostics tab!")
    st.markdown("#### This tab will provide diagnostics-level information, including historical viral load test records, number of VL tests and recent VL test results.")

    if st.session_state.search_results:
        joint = st.session_state.search_results

        df5['mpi_ref'] = df5['mpi_ref'].astype(str)
        df5['joint_id'] = df5['mpi_ref'] + '_' + df5['EMR_ID'].astype(str)
        # Continue processing if joint ID is found
        df5['DateTest'] = pd.to_datetime(df5['DateTest'])
        diagnostic_records = df5[df5.joint_id == joint].sort_values(by='DateTest')[['DateTest', 'Resultat']].reset_index(drop=True)
        diagnostic_records['DateTest'] = diagnostic_records['DateTest'].dt.date

        # Container for displaying data
        dash_1 = st.container()

        with dash_1:
            col1, col2 = st.columns(2)
            with col1:
                # Display the diagnostic records in a pretty table
                st.markdown("##### Your Viral Load Test Records:")
                st.dataframe(diagnostic_records)

                # Extract Number of times got HIV test
                num_hiv = diagnostic_records.shape[0]

                # Check if Number of times got HIV test is found and display
                st.markdown(f"##### Number of times you've got a Viral Load test: {num_hiv}")
            
            with col2:
                # Extract whether Test within the last year
                latest_test_date = diagnostic_records['DateTest'].max()
                if pd.notna(latest_test_date):
                    one_year_ago = datetime.now() - timedelta(days=365)
                    within_one_year = 'Yes' if latest_test_date >= one_year_ago else 'No'
                    st.markdown(f"##### Have you had a Viral Load test within the last year: {within_one_year}")
                else:
                    st.markdown("##### Have you had a Viral Load test within the last year: Unknown")

                # Extract recent Viral Load Test result
                last_VL_result = diagnostic_records['Resultat'].iloc[-1] if not diagnostic_records.empty else 'Unknown'
                st.markdown(f"##### Recent Viral Load Test Result: {last_VL_result}")

    else:
        st.markdown("#### No matching records found for the given EMR ID and institution.")






css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:18px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)



