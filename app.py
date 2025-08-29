# app.py
# This is the main script for the Streamlit web application.

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# Import the machine learning functions from your ml_module
from ml_module import detect_hotspots_dbscan, detect_hotspots_kmeans

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CrimeLens Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# --- HELPER FUNCTIONS ---
@st.cache_data # Use Streamlit's caching to avoid reloading data on every interaction
def load_data(file):
    """
    Loads data from an uploaded file (CSV or Excel) and performs basic cleaning.
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Standardize column names (e.g., "Crime Type" -> "crime_type")
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # --- IMPORTANT ---
        # You MUST have columns named 'latitude', 'longitude', and a datetime column.
        # Adjust the column names below to match your dataset EXACTLY.
        required_cols = ['latitude', 'longitude', 'crime_type', 'date/time'] # Example column names
        
        # Check if required columns exist
        if not all(col in df.columns for col in required_cols):
            st.error(f"Error: Your file must contain the following columns: {', '.join(required_cols)}")
            return None

        # Drop rows with missing geo-coordinates
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        # Convert date column to datetime objects
        df['datetime'] = pd.to_datetime(df['date/time'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True) # Drop rows where date conversion failed

        return df
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
        return None

@st.cache_data
def convert_df_to_csv(df):
    """
    Converts a DataFrame to a CSV string for downloading.
    """
    return df.to_csv(index=False).encode('utf-8')

# --- MAIN APP LAYOUT ---
st.title("🗺️ CrimeLens: Predictive Crime Analysis Dashboard")
st.markdown("Upload your dataset to visualize crime patterns and detect hotspots.")

# --- SIDEBAR ---
st.sidebar.title("Controls & Filters")
uploaded_file = st.sidebar.file_uploader(
    "Upload your crime dataset",
    type=["csv", "excel"]
)

# --- CONDITIONAL CONTENT (shows only after file upload) ---
if uploaded_file is not None:
    crime_data = load_data(uploaded_file)

    if crime_data is not None:
        st.sidebar.success("File uploaded and processed successfully!")
        
        # --- SIDEBAR FILTERS ---
        st.sidebar.header("Filter Data")
        
        # Filter by Crime Type
        crime_types = sorted(crime_data['crime_type'].unique())
        selected_crime_types = st.sidebar.multiselect(
            "Select Crime Type(s)", 
            crime_types, 
            default=crime_types[:5] # Default to first 5 types
        )

        # Filter by Date Range
        min_date = crime_data['datetime'].min().date()
        max_date = crime_data['datetime'].max().date()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply filters
        start_date, end_date = date_range
        filtered_data = crime_data[
            (crime_data['crime_type'].isin(selected_crime_types)) &
            (crime_data['datetime'].dt.date >= start_date) &
            (crime_data['datetime'].dt.date <= end_date)
        ]

        st.header("Filtered Crime Data")
        st.dataframe(filtered_data.head())

        # --- MAP VISUALIZATION ---
        st.header("Interactive Crime Map")
        if not filtered_data.empty:
            map_center = [filtered_data['latitude'].mean(), filtered_data['longitude'].mean()]
            crime_map = folium.Map(location=map_center, zoom_start=12)

            # Add a heatmap layer
            heat_data = [[row['latitude'], row['longitude']] for index, row in filtered_data.iterrows()]
            HeatMap(heat_data).add_to(crime_map)
            
            st_folium(crime_map, width=725, height=500, use_container_width=True)
        else:
            st.warning("No data matches the current filter settings.")

        # --- ANALYSIS CONTROLS ---
        st.sidebar.header("Analysis Controls")
        analysis_type = st.sidebar.radio("Select Analysis Type", ["DBSCAN", "K-Means"])

        run_analysis = st.sidebar.button("Run Hotspot Analysis")

        if run_analysis and not filtered_data.empty:
            with st.spinner("Analyzing hotspots... This may take a moment."):
                if analysis_type == "DBSCAN":
                    eps = st.sidebar.slider("DBSCAN Epsilon (distance)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
                    min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 20, 5)
                    hotspot_data = detect_hotspots_dbscan(filtered_data.copy(), eps=eps, min_samples=min_samples)
                else: # K-Means
                    n_clusters = st.sidebar.slider("Number of Clusters (K-Means)", 2, 20, 5)
                    hotspot_data = detect_hotspots_kmeans(filtered_data.copy(), n_clusters=n_clusters)

            st.header("Hotspot Analysis Results")
            # Filter out noise points for better visualization (-1 is noise in DBSCAN)
            hotspot_clusters = hotspot_data[hotspot_data['cluster'] != -1]
            st.dataframe(hotspot_clusters)
            
            # Visualize clusters on a map
            if not hotspot_clusters.empty:
                hotspot_map = folium.Map(location=map_center, zoom_start=12)
                
                # Create a color map for clusters
                colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
                
                for _, row in hotspot_clusters.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color=colors[row['cluster'] % len(colors)],
                        fill=True,
                        fill_color=colors[row['cluster'] % len(colors)]
                    ).add_to(hotspot_map)
                
                st_folium(hotspot_map, use_container_width=True)
            else:
                st.warning("No distinct hotspots were detected with the current parameters.")
        
        elif run_analysis and filtered_data.empty:
            st.error("Cannot run analysis because there is no data matching the current filters.")

        # --- DATA EXPORT ---
        st.sidebar.header("Export Data")
        csv_export = convert_df_to_csv(filtered_data)
        st.sidebar.download_button(
            label="Download Filtered Data as CSV",
            data=csv_export,
            file_name='filtered_crime_data.csv',
            mime='text/csv',
        )

else:
    st.info("Awaiting for a file to be uploaded. Please use the sidebar to begin.")

