# app.py
# Final Corrected Version: Fixes the state-reset loop on button clicks.

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import os

# Import our modules
import db
from ml_module import detect_hotspots_dbscan, detect_hotspots_kmeans, forecast_hotspot_trends
from report_generator import create_report

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CrimeLens Dashboard", page_icon="⚖️", layout="wide")

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(file):
    # (This function is correct and remains unchanged)
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

# --- AUTHENTICATION & UI ---
if 'username' not in st.session_state: st.session_state.username = None

with st.sidebar:
    # (Authentication logic is correct and remains unchanged)
    if st.session_state.username:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    else:
        choice = option_menu("Login / Sign Up", ["Login", "Sign Up"], icons=['box-arrow-in-right', 'person-plus'], menu_icon="app-indicator")
        if choice == "Login":
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if db.login(username, password):
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        elif choice == "Sign Up":
            st.subheader("Create a New Account")
            new_username = st.text_input("Choose a Username", key="signup_user")
            new_password = st.text_input("Choose a Password", type="password", key="signup_pass")
            if st.button("Sign Up"):
                success, message = db.sign_up(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

# --- VIEW-SPECIFIC FUNCTIONS ---

def main_dashboard_view(crime_data):
    """Contains all logic for the main, single-panel dashboard."""
    st.title("🔮 CrimeLens: Predictive Crime Analysis & Forecasting")
    st.markdown("Use the filters in the sidebar to visualize patterns, detect hotspots, and forecast future trends.")

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
            st_folium(crime_map, use_container_width=True, key="main_map")
        else:
            st.warning("No data matches the current filter settings.")

        st.sidebar.header("Analysis Controls")
        analysis_type = st.sidebar.radio("Select Analysis Type", ["DBSCAN", "K-Means"])
        if analysis_type == "DBSCAN":
            eps = st.sidebar.slider("DBSCAN Epsilon", 0.001, 0.1, 0.01, 0.001, format="%.3f")
            min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 20, 5)
        else:
            n_clusters = st.sidebar.slider("Number of Clusters (K-Means)", 2, 20, 5)

        if st.sidebar.button("Run Hotspot Analysis"):
            if not filtered_data.empty:
                with st.spinner("Analyzing hotspots..."):
                    if analysis_type == "DBSCAN":
                        hotspot_data = detect_hotspots_dbscan(filtered_data.copy(), eps=eps, min_samples=min_samples)
                    else:
                        hotspot_data = detect_hotspots_kmeans(filtered_data.copy(), n_clusters=n_clusters)
                    
                    st.session_state.hotspot_clusters = hotspot_data[hotspot_data['cluster'] != -1]
                    st.session_state.pdf_data = None
        
        st.divider()
        st.header("📊 Temporal Analysis")
        if not filtered_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                hourly_data = filtered_data['datetime'].dt.hour.value_counts().sort_index()
                fig_hourly = px.bar(hourly_data, x=hourly_data.index, y=hourly_data.values, labels={'index': 'Hour of Day', 'y': 'Number of Crimes'}, title="Crimes by Hour of Day")
                st.plotly_chart(fig_hourly, use_container_width=True)
            with col2:
                daily_data = filtered_data['datetime'].dt.day_name().value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                fig_daily = px.bar(daily_data, x=daily_data.index, y=daily_data.values, labels={'index': 'Day of Week', 'y': 'Number of Crimes'}, title="Crimes by Day of Week")
                st.plotly_chart(fig_daily, use_container_width=True)
        
        st.divider()
        st.header("🔮 Predictive Forecasting")
        if st.session_state.get('hotspot_clusters') is None:
            st.warning("Please run a hotspot analysis to enable predictive forecasting.")
        elif st.session_state.hotspot_clusters.empty:
            st.warning("Analysis complete, but no hotspots were found with the current parameters. Please adjust the sliders in the sidebar (e.g., increase Epsilon) and run the analysis again.")
        else:
            cluster_options = sorted(st.session_state.hotspot_clusters['cluster'].unique())
            selected_cluster = st.selectbox("Select Hotspot Cluster ID", options=cluster_options)
            forecast_days = st.slider("Days to Forecast", 7, 90, 30)
            if st.button(f"Forecast Trends for Hotspot {selected_cluster}"):
                with st.spinner(f"Training model..."):
                    history_df, forecast_df = forecast_hotspot_trends(st.session_state.hotspot_clusters, selected_cluster, forecast_days)
                if forecast_df is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=history_df['datetime'], y=history_df['crime_count'], mode='lines', name='Historical'))
                    fig.add_trace(go.Scatter(x=forecast_df['index'], y=forecast_df['forecast'], mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=forecast_df['index'].tolist() + forecast_df['index'].tolist()[::-1], y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1], fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not generate forecast.")
        
        st.divider()
        st.header("📁 Reporting & Exports")
        if st.session_state.get('hotspot_clusters') is None:
            st.warning("Please run a hotspot analysis to enable reporting and exports.")
        elif st.session_state.hotspot_clusters.empty:
            st.warning("Analysis complete, but no hotspots were found with the current parameters. Reporting is disabled.")
        else:
            if st.button("Generate PDF Report"):
                with st.spinner("Generating Report... This may take a moment."):
                    hourly_data = filtered_data['datetime'].dt.hour.value_counts().sort_index()
                    fig_hourly = px.bar(hourly_data, x=hourly_data.index, y=hourly_data.values, labels={'index': 'Hour of Day', 'y': 'Number of Crimes'})
                    
                    daily_data = filtered_data['datetime'].dt.day_name().value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                    fig_daily = px.bar(daily_data, x=daily_data.index, y=daily_data.values, labels={'index': 'Day of Week', 'y': 'Number of Crimes'})
                    
                    fig_hourly.write_image("temp_hourly_chart.png", scale=2)
                    fig_daily.write_image("temp_daily_chart.png", scale=2)
                    
                    report_data = create_report(
                        filtered_data=filtered_data,
                        hotspot_data=st.session_state.hotspot_clusters,
                        date_range=(start_date, end_date),
                        crime_types=selected_crime_types,
                        hourly_chart_path="temp_hourly_chart.png",
                        daily_chart_path="temp_daily_chart.png"
                    )
                    
                    st.session_state.pdf_data = report_data
                    os.remove("temp_hourly_chart.png")
                    os.remove("temp_daily_chart.png")

            if st.session_state.get('pdf_data'):
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.pdf_data,
                    file_name=f"CrimeLens_Report_{start_date}_to_{end_date}.pdf",
                    mime="application/pdf"
                )

def comparative_analysis_view(crime_data):
    """Contains all logic for the side-by-side comparative view."""
    # (This function is correct and remains unchanged)
    st.header("⚖️ Comparative Analysis View")
    st.markdown("Analyze two scenarios side-by-side. Use the filters in each panel to compare crime types or date ranges.")
    
    col1, col2 = st.columns(2)
    crime_types = sorted(crime_data['crime_type'].unique())
    min_date, max_date = crime_data['datetime'].min().date(), crime_data['datetime'].max().date()

    with col1:
        st.subheader("Panel A")
        date_range_A = st.date_input("Select Date Range (A)", [min_date, max_date], key="date_A")
        selected_crimes_A = st.multiselect("Select Crime Type(s) (A)", crime_types, default=crime_types[:1], key="crimes_A")
        if len(date_range_A) == 2:
            start_A, end_A = date_range_A
            filtered_A = crime_data[(crime_data['crime_type'].isin(selected_crimes_A)) & (crime_data['datetime'].dt.date >= start_A) & (crime_data['datetime'].dt.date <= end_A)]
            st.metric("Total Crimes (A)", len(filtered_A))
            if not filtered_A.empty:
                map_A = folium.Map(location=[filtered_A['latitude'].mean(), filtered_A['longitude'].mean()], zoom_start=12)
                HeatMap([[r['latitude'], r['longitude']] for i, r in filtered_A.iterrows()]).add_to(map_A)
                st_folium(map_A, use_container_width=True, key="map_A")
            else:
                st.warning("No data for selection A.")

    with col2:
        st.subheader("Panel B")
        date_range_B = st.date_input("Select Date Range (B)", [min_date, max_date], key="date_B")
        selected_crimes_B = st.multiselect("Select Crime Type(s) (B)", crime_types, default=crime_types[1:2] if len(crime_types) > 1 else crime_types, key="crimes_B")
        if len(date_range_B) == 2:
            start_B, end_B = date_range_B
            filtered_B = crime_data[(crime_data['crime_type'].isin(selected_crimes_B)) & (crime_data['datetime'].dt.date >= start_B) & (crime_data['datetime'].dt.date <= end_B)]
            st.metric("Total Crimes (B)", len(filtered_B))
            if not filtered_B.empty:
                map_B = folium.Map(location=[filtered_B['latitude'].mean(), filtered_B['longitude'].mean()], zoom_start=12)
                HeatMap([[r['latitude'], r['longitude']] for i, r in filtered_B.iterrows()]).add_to(map_B)
                st_folium(map_B, use_container_width=True, key="map_B")
            else:
                st.warning("No data for selection B.")

# --- MAIN CONTROLLER ---
if st.session_state.username:
    
    # Initialize all session state variables
    if 'crime_data' not in st.session_state: st.session_state.crime_data = None
    if 'hotspot_clusters' not in st.session_state: st.session_state.hotspot_clusters = None
    if 'pdf_data' not in st.session_state: st.session_state.pdf_data = None
    if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None

    with st.sidebar:
        st.sidebar.divider()
        main_view = option_menu("Main Menu", ["Main Dashboard", "Comparative Analysis"], icons=['clipboard-data', 'layout-split'], menu_icon="cast", default_index=0)
        st.sidebar.divider()
    
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # --- THIS IS THE FIX ---
    # This logic now only runs if the uploaded file is new.
    if uploaded_file and (st.session_state.uploaded_file_name != uploaded_file.name):
        st.session_state.crime_data = load_data(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.hotspot_clusters = None # Reset analysis for new file
        st.session_state.pdf_data = None

    if st.session_state.crime_data is not None:
        if main_view == "Main Dashboard":
            main_dashboard_view(st.session_state.crime_data)
        elif main_view == "Comparative Analysis":
            comparative_analysis_view(st.session_state.crime_data)
    else:
        st.info("Please upload a crime dataset using the sidebar to begin analysis.")
else:
    st.warning("Please Login or Sign Up to use the application.")

