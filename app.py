import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from streamlit_option_menu import option_menu
import plotly.express as px
from datetime import datetime

# Import our database, ml, and new report generator modules
import db
from ml_module import detect_hotspots_dbscan, detect_hotspots_kmeans, analyze_hotspot_trends
from report_generator import create_report # ✨ NEW IMPORT

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CrimeLens Dashboard", page_icon="🚨", layout="wide")

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        required_cols = ['latitude', 'longitude', 'crime_type', 'date/time']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Error: Your file must contain the following columns: {', '.join(required_cols)}")
            return None
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        df['datetime'] = pd.to_datetime(df['date/time'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        return None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- STATIC DATA: EMERGENCY CONTACTS ---
emergency_contacts = {
    "Select a City": {}, "Generic (India)": {"Police": "100", "Ambulance": "102", "Fire": "101", "National Emergency Number": "112"},
    "New York (USA)": {"Emergency": "911", "Non-Emergency Police": "311", "Crime Stoppers": "1-800-577-TIPS"},
    "London (UK)": {"Emergency": "999", "Non-Emergency Police": "101", "Crime Stoppers": "0800 555 111"},
    "Mumbai (India)": {"Police Control Room": "100", "Traffic Police": "103", "Anti-Terror Helpline": "1090"}
}

# --- AUTHENTICATION & UI ---
if 'username' not in st.session_state:
    st.session_state.username = None

with st.sidebar:
    if st.session_state.username:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.username = None
            st.session_state.hotspot_clusters = None
            st.rerun()
    else:
        choice = option_menu("Login / Sign Up", ["Login", "Sign Up"], icons=['box-arrow-in-right', 'person-plus'], menu_icon="app-indicator")
        if choice == "Login":
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if not username or not password:
                    st.error("Username and Password are required.")
                elif db.login(username, password):
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        elif choice == "Sign Up":
            st.subheader("Create a New Account")
            new_username = st.text_input("Choose a Username", key="signup_user")
            new_password = st.text_input("Choose a Password", type="password", key="signup_pass")
            if st.button("Sign Up"):
                if not new_username or not new_password:
                    st.error("Username and Password are required.")
                elif db.sign_up(new_username, new_password):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username already exists or an error occurred.")

# --- MAIN APPLICATION LOGIC ---
if st.session_state.username:
    st.title("🚨 CrimeLens: Predictive Crime Analysis & Alert System")
    st.markdown("Upload a dataset to visualize crime patterns, detect hotspots, and generate reports.")

    if 'hotspot_clusters' not in st.session_state:
        st.session_state.hotspot_clusters = None

    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if not uploaded_file.name.lower().endswith(('.csv', '.xlsx', '.xls')):
            st.error("Invalid file type. Please upload a CSV or Excel file.")
        else:
            db.log_activity(st.session_state.username, "File Upload", {"filename": uploaded_file.name})
            crime_data = load_data(uploaded_file)
            if crime_data is not None:
                st.sidebar.header("Filter Data")
                crime_types = sorted(crime_data['crime_type'].unique())
                selected_crime_types = st.sidebar.multiselect("Select Crime Type(s)", crime_types, default=crime_types[:5])
                min_date, max_date = crime_data['datetime'].min().date(), crime_data['datetime'].max().date()
                date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_data = crime_data[
                        (crime_data['crime_type'].isin(selected_crime_types)) & 
                        (crime_data['datetime'].dt.date >= start_date) & 
                        (crime_data['datetime'].dt.date <= end_date)
                    ]

                    st.header("Interactive Crime Map")
                    if not filtered_data.empty:
                        map_center = [filtered_data['latitude'].mean(), filtered_data['longitude'].mean()]
                        crime_map = folium.Map(location=map_center, zoom_start=12)
                        HeatMap([[row['latitude'], row['longitude']] for _, row in filtered_data.iterrows()]).add_to(crime_map)
                        st_folium(crime_map, use_container_width=True)
                    else:
                        st.warning("No data matches the current filter settings.")

                    # ✨ NEW FEATURE: TEMPORAL ANALYSIS DASHBOARD ✨
                    st.divider()
                    st.header("📊 Temporal Analysis")
                    if not filtered_data.empty:
                        filtered_data['hour'] = filtered_data['datetime'].dt.hour
                        hourly_crimes = filtered_data.groupby('hour').size().reset_index(name='count')
                        fig_hourly = px.bar(hourly_crimes, x='hour', y='count', title='Crime Count by Hour of Day', labels={'hour': 'Hour of Day', 'count': 'Number of Crimes'})
                        
                        filtered_data['day_of_week'] = filtered_data['datetime'].dt.day_name()
                        daily_crimes = filtered_data.groupby('day_of_week').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index(name='count')
                        fig_daily = px.bar(daily_crimes, x='day_of_week', y='count', title='Crime Count by Day of Week', labels={'day_of_week': 'Day of Week', 'count': 'Number of Crimes'})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_hourly, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_daily, use_container_width=True)
                    else:
                        st.info("Not enough data to display temporal analysis.")
                    st.divider()

                    st.sidebar.header("Analysis Controls")
                    analysis_type = st.sidebar.radio("Select Analysis Type", ["DBSCAN", "K-Means"])
                    if st.sidebar.button("Run Hotspot Analysis"):
                        if not filtered_data.empty:
                            with st.spinner("Analyzing hotspots..."):
                                if analysis_type == "DBSCAN":
                                    eps = st.sidebar.slider("DBSCAN Epsilon", 0.001, 0.1, 0.01, 0.001, format="%.3f")
                                    min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 20, 5)
                                    hotspot_data = detect_hotspots_dbscan(filtered_data.copy(), eps=eps, min_samples=min_samples)
                                else: # K-Means
                                    n_clusters = st.sidebar.slider("Number of Clusters (K-Means)", 2, 20, 5)
                                    hotspot_data = detect_hotspots_kmeans(filtered_data.copy(), n_clusters=n_clusters)
                                
                                st.session_state.hotspot_clusters = hotspot_data[hotspot_data['cluster'] != -1]
                                st.rerun()

                    if st.session_state.hotspot_clusters is not None and not st.session_state.hotspot_clusters.empty:
                        st.header("Hotspot Analysis Results")
                        st.dataframe(st.session_state.hotspot_clusters)
                        
                        st.header("🚨 Alert Prediction System")
                        # (Alert System UI code remains here...)
                        
                        # ✨ NEW FEATURE: REPORTING & EXPORTS ✨
                        st.divider()
                        st.header("📁 Reporting & Exports")
                        st.markdown("Download the raw hotspot data or generate a comprehensive PDF report of your analysis.")
                        
                        col1_rep, col2_rep = st.columns(2)
                        with col1_rep:
                            st.download_button(
                                label="Download Data as CSV",
                                data=convert_df_to_csv(st.session_state.hotspot_clusters),
                                file_name=f"hotspot_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                help="Download the clustered hotspot data in CSV format."
                            )

                        with col2_rep:
                            if 'pdf_generated' not in st.session_state:
                                st.session_state.pdf_generated = False
                            if 'pdf_data' not in st.session_state:
                                st.session_state.pdf_data = None

                            if st.button("Generate PDF Report", help="Create a detailed PDF summary of the current analysis."):
                                with st.spinner("Generating your report... Please wait."):
                                    fig_hourly.write_image("temp_hourly_chart.png", scale=2)
                                    fig_daily.write_image("temp_daily_chart.png", scale=2)
                                    
                                    st.session_state.pdf_data = create_report(
                                        filtered_data=filtered_data,
                                        hotspot_data=st.session_state.hotspot_clusters,
                                        date_range=date_range,
                                        crime_types=selected_crime_types,
                                        hourly_chart_path="temp_hourly_chart.png",
                                        daily_chart_path="temp_daily_chart.png"
                                    )
                                    st.session_state.pdf_generated = True
                            
                            if st.session_state.pdf_generated:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=st.session_state.pdf_data,
                                    file_name=f"CrimeLens_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf"
                                )
                        st.divider()
else:
    st.warning("Please Login or Sign Up to use the application.")
