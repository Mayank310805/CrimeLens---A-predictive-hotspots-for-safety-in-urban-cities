# app.py
# Refactored to make all analysis sections visible immediately after file upload.

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
from ml_module import detect_hotspots_dbscan, detect_hotspots_kmeans, analyze_hotspot_trends, forecast_hotspot_trends
from report_generator import create_report

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CrimeLens Dashboard", page_icon="🔮", layout="wide")

# --- HELPER FUNCTIONS (No changes needed here) ---
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

# --- AUTHENTICATION & UI (No changes needed here) ---
if 'username' not in st.session_state: st.session_state.username = None

with st.sidebar:
    if st.session_state.username:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.username = None
            st.session_state.hotspot_clusters = None
            st.session_state.pdf_data = None # Clear pdf data on logout
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
    if 'pdf_data' not in st.session_state:
        st.session_state.pdf_data = None

    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        db.log_activity(st.session_state.username, "File Upload", {"filename": uploaded_file.name})
        crime_data = load_data(uploaded_file)
        if crime_data is not None:
            st.sidebar.header("Filter Data")
            # (Filtering logic remains the same)
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
                # (Map logic remains the same)
                if not filtered_data.empty:
                    map_center = [filtered_data['latitude'].mean(), filtered_data['longitude'].mean()]
                    crime_map = folium.Map(location=map_center, zoom_start=12)
                    HeatMap([[row['latitude'], row['longitude']] for _, row in filtered_data.iterrows()]).add_to(crime_map)
                    st_folium(crime_map, use_container_width=True)
                else:
                    st.warning("No data matches the current filter settings.")

                st.sidebar.header("Analysis Controls")
                # (Analysis controls remain the same)
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
                            st.session_state.pdf_data = None # Clear old report data
                            st.rerun()

                # --- ✨ RESTRUCTURED SECTIONS START HERE ✨ ---
                
                # SECTION 1: TEMPORAL ANALYSIS (Always shows with filtered data)
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
                else:
                    st.info("No data available for temporal analysis based on current filters.")

                # SECTION 2: PREDICTIVE FORECASTING
                st.divider()
                st.header("🔮 Predictive Forecasting")
                st.markdown("Select a detected hotspot to forecast its future crime trends.")

                if st.session_state.hotspot_clusters is not None and not st.session_state.hotspot_clusters.empty:
                    # This part runs ONLY if hotspots have been detected
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
                        # (Forecasting logic remains the same)
                        with st.spinner(f"Training model..."):
                            history_df, forecast_df = forecast_hotspot_trends(st.session_state.hotspot_clusters, selected_cluster, forecast_days)
                        if forecast_df is not None:
                            st.success("Forecast generated!")
                            # (Chart logic remains the same)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=history_df['datetime'], y=history_df['crime_count'], mode='lines', name='Historical'))
                            fig.add_trace(go.Scatter(x=forecast_df['index'], y=forecast_df['forecast'], mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=forecast_df['index'].tolist() + forecast_df['index'].tolist()[::-1], y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1], fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
                            fig.update_layout(title=f"Crime Forecast for Hotspot {selected_cluster}", xaxis_title="Date", yaxis_title="Number of Crimes")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Could not generate forecast. Insufficient historical data (min. 14 days) for this hotspot in the selected range.")
                else:
                    st.warning("Please run a hotspot analysis first to enable predictive forecasting.")

                # SECTION 3: REPORTING & EXPORTS
                st.divider()
                st.header("📁 Reporting & Exports")
                if st.session_state.hotspot_clusters is not None and not st.session_state.hotspot_clusters.empty:
                    # This part runs ONLY if hotspots have been detected
                    st.markdown("Download the raw hotspot data or generate a comprehensive PDF summary of your analysis.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download Hotspot Data (CSV)",
                            data=convert_df_to_csv(st.session_state.hotspot_clusters),
                            file_name="hotspot_data.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        if st.button("Generate PDF Report"):
                            with st.spinner("Generating Report..."):
                                # (Report generation logic remains the same)
                                hourly_counts = filtered_data['datetime'].dt.hour.value_counts().sort_index()
                                fig_hourly = px.bar(hourly_counts, x=hourly_counts.index, y=hourly_counts.values)
                                daily_counts = filtered_data['datetime'].dt.day_name().value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                                fig_daily = px.bar(daily_counts, x=daily_counts.index, y=daily_counts.values)
                                fig_hourly.write_image("temp_hourly_chart.png", scale=2)
                                fig_daily.write_image("temp_daily_chart.png", scale=2)
                                report_data = create_report(filtered_data, st.session_state.hotspot_clusters, (start_date, end_date), selected_crime_types, "temp_hourly_chart.png", "temp_daily_chart.png")
                                st.session_state.pdf_data = report_data
                                os.remove("temp_hourly_chart.png")
                                os.remove("temp_daily_chart.png")
                    
                    if st.session_state.pdf_data:
                        st.download_button(
                            label="Download PDF Report",
                            data=st.session_state.pdf_data,
                            file_name=f"CrimeLens_Report_{start_date}_to_{end_date}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.warning("Please run a hotspot analysis first to enable reporting and exports.")

else:
    st.warning("Please Login or Sign Up to use the application.")

