import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import altair as alt 
import numpy as np

# Import the dataset
image = "CGHPI.png"


# Streamlit application
def app():
    # Main page content
    st.set_page_config(page_title = 'Haiti Prediction Dashboard', page_icon='🇭🇹',layout='wide')

    # Use columns for side-by-side layout
    col1, col2 = st.columns([1, 3])  # Adjust the width ratio as needed

    # Place the image and title in the columns
    with col1:
        st.image(image, width=230)

    with col2:
        st.title('🇭🇹  Haiti Prediction Dashboard')

                            
    st.markdown("""
    This dashboard is specifically designed to predict treatment outcomes for HIV patients within the next 28 days, using their current clinical records and personal information. By estimating the likelihood of patients entering PIT (Out of care) status, healthcare providers in Haiti can intervene promptly if necessary. However, since some clinical data may be inaccurately recorded due to manual entry processes, one tab allows patients to verify and confirm their records before predictions are made. Another tab is dedicated to assisting healthcare providers in monitoring the tool's usage regularly, ensuring that all interventions are based on accurate and up-to-date information. Below are details on the three sub-tabs, each tailored to support these critical functionalities efficiently.

    1. **Tab 1: Patient Information** - Search, check, and verify patient records.
    2. **Tab 2: Predictive Analytics** - Generate predictive outcomes for patient care.
    3. **Tab 3: Results Monitoring** - Track prediction outcomes by day and institution.

    Feel free to explore any tab to interact with the dashboard!
    """)





if __name__ == "__main__":
    app()
