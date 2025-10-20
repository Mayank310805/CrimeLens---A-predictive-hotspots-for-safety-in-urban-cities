# app.py
# Updated main script with a predictive forecasting feature for hotspots.

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import os

# Import our updated modules
import db
from ml_module import detect_hotspots_dbscan, detect_hotspots_kmeans, analyze_hotspot_trends, forecast_hotspot_trends
from report_generator import create_report

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CrimeLens Dashboard", page_icon="🔮", layout="wide")

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

emergency_contacts = {
    "Select a City": {}, "Generic (India)": {"Police": "100", "Ambulance": "102", "Fire": "101", "National Emergency Number": "112"},
    "New York (USA)": {"Emergency": "911", "Non-Emergency Police": "311", "Crime Stoppers": "1-800-577-TIPS"},
    "London (UK)": {"Emergency": "999", "Non-Emergency Police": "101", "Crime Stoppers": "0800 555 111"},
    "Mumbai (India)": {"Police Control Room": "100", "Traffic Police": "103", "Anti-Terror Helpline": "1090"}
}

# --- AUTHENTICATION & UI ---
if 'username' not in st.session_state: st.session_state.username = None

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

# --- MAIN APPLICATION LOGIC ---
if st.session_state.username:
    st.title("🔮 CrimeLens: Predictive Crime Analysis & Forecasting")
    st.markdown("Upload a dataset to visualize patterns, detect hotspots, and forecast future trends.")

    if 'hotspot_clusters' not in st.session_state:
        st.session_state.hotspot_clusters = None

    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
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

                st.sidebar.header("Analysis Controls")
                analysis_type = st.sidebar.radio("Select Analysis Type", ["DBSCAN", "K-Means"])
                
                if analysis_type == "DBSCAN":
                    eps = st.sidebar.slider("DBSCAN Epsilon", 0.001, 0.1, 0.01, 0.001, format="%.3f")
                    min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 20, 5)
                else: # K-Means
                    n_clusters = st.sidebar.slider("Number of Clusters (K-Means)", 2, 20, 5)

                if st.sidebar.button("Run Hotspot Analysis"):
                    if not filtered_data.empty:
                        with st.spinner("Analyzing hotspots..."):
                            if analysis_type == "DBSCAN":
                                hotspot_data = detect_hotspots_dbscan(filtered_data.copy(), eps=eps, min_samples=min_samples)
                            else:
                                hotspot_data = detect_hotspots_kmeans(filtered_data.copy(), n_clusters=n_clusters)
                            
                            st.session_state.hotspot_clusters = hotspot_data[hotspot_data['cluster'] != -1]
                            st.rerun()
                
                if st.session_state.hotspot_clusters is not None and not st.session_state.hotspot_clusters.empty:
                    st.divider()
                    st.header("🔮 Predictive Forecasting")
                    st.markdown("Select a detected hotspot to forecast its future crime trends.")

                    cluster_options = sorted(st.session_state.hotspot_clusters['cluster'].unique())
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        selected_cluster = st.selectbox("Select Hotspot Cluster ID", options=cluster_options)
                        forecast_days = st.slider("Days to Forecast", 7, 90, 30)
                    
                    with col2:
                        st.info(f"""
                        **How this works:**
                        1.  We isolate all crimes from **Hotspot {selected_cluster}**.
                        2.  A SARIMA time-series model is trained on its daily crime counts.
                        3.  The model then predicts the trend for the next **{forecast_days} days**.
                        """)

                    if st.button(f"Forecast Trends for Hotspot {selected_cluster}"):
                        with st.spinner(f"Training model and forecasting for Hotspot {selected_cluster}..."):
                            history_df, forecast_df = forecast_hotspot_trends(
                                st.session_state.hotspot_clusters, selected_cluster, forecast_days
                            )
                        
                        if forecast_df is not None:
                            st.success(f"Forecast generated successfully for Hotspot {selected_cluster}!")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=history_df['datetime'], y=history_df['crime_count'], mode='lines', name='Historical Daily Crimes'))
                            fig.add_trace(go.Scatter(x=forecast_df['index'], y=forecast_df['forecast'], mode='lines', name='Forecasted Crimes', line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=forecast_df['index'].tolist() + forecast_df['index'].tolist()[::-1], y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1], fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='Confidence Interval'))
                            fig.update_layout(title=f"Crime Forecast for Hotspot {selected_cluster}", xaxis_title="Date", yaxis_title="Number of Crimes", legend_title="Legend")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Could not generate a forecast for Hotspot {selected_cluster}. This usually happens if there is not enough historical data (at least 14 days) for this specific hotspot in the selected time range.")

                    st.divider()
                    st.header("📊 Temporal Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        hourly_data = filtered_data['datetime'].dt.hour
                        hourly_counts = hourly_data.value_counts().sort_index()
                        fig_hourly = px.bar(x=hourly_counts.index, y=hourly_counts.values, labels={'x': 'Hour of Day', 'y': 'Number of Crimes'}, title="Crimes by Hour of Day")
                        st.plotly_chart(fig_hourly, use_container_width=True)

                    with col2:
                        daily_data = filtered_data['datetime'].dt.day_name()
                        daily_counts = daily_data.value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                        fig_daily = px.bar(x=daily_counts.index, y=daily_counts.values, labels={'x': 'Day of Week', 'y': 'Number of Crimes'}, title="Crimes by Day of Week")
                        st.plotly_chart(fig_daily, use_container_width=True)

                    st.divider()
                    st.header("📁 Reporting & Exports")
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating Report... This may take a moment."):
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

                    if 'pdf_data' in st.session_state and st.session_else:
                        st.warning("Please Login or Sign Up to use the application.")

