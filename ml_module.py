# ml_module.py
# Updated with a new function for time-series forecasting.

from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
from datetime import timedelta
import statsmodels.api as sm

def detect_hotspots_dbscan(data, eps=0.01, min_samples=5):
    """ Detects crime hotspots using the DBSCAN algorithm. """
    coords = data[['latitude', 'longitude']].values
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    data['cluster'] = db.labels_
    return data

def detect_hotspots_kmeans(data, n_clusters=5):
    """ Detects crime hotspots using the K-Means algorithm. """
    coords = data[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(coords)
    data['cluster'] = kmeans.labels_
    return data

def analyze_hotspot_trends(hotspot_data, time_window_days):
    """ Analyzes recent crime activity within detected hotspots. """
    recent_cutoff = pd.Timestamp.now() - timedelta(days=time_window_days)
    recent_crimes = hotspot_data[hotspot_data['datetime'] >= recent_cutoff]
    if recent_crimes.empty:
        return pd.DataFrame(columns=['cluster', 'recent_crime_count', 'latitude', 'longitude'])
    alert_summary = recent_crimes.groupby('cluster').size().reset_index(name='recent_crime_count')
    centroids = hotspot_data.groupby('cluster')[['latitude', 'longitude']].mean().reset_index()
    return pd.merge(alert_summary, centroids, on='cluster')

# ✨ --- NEW PREDICTIVE FORECASTING FUNCTION --- ✨
def forecast_hotspot_trends(hotspot_data, cluster_id, forecast_days):
    """
    Trains a SARIMA model and forecasts future crime trends for a specific hotspot.

    Args:
        hotspot_data (pd.DataFrame): The full dataframe with cluster assignments.
        cluster_id (int): The specific cluster ID to forecast.
        forecast_days (int): The number of days into the future to forecast.

    Returns:
        tuple: A tuple containing (historical_data, forecast_data) DataFrames.
               Returns (None, None) if data is insufficient for modeling.
    """
    # Filter data for the selected hotspot and create a time series
    cluster_data = hotspot_data[hotspot_data['cluster'] == cluster_id].copy()
    cluster_data.set_index('datetime', inplace=True)
    
    # Resample to get daily crime counts. Fill missing days with 0.
    daily_counts = cluster_data.resample('D').size().rename('crime_count')
    
    # A minimum number of data points are needed for a reliable forecast
    if len(daily_counts) < 14: # Need at least two weeks of data
        return None, None

    # Train a SARIMA model. The parameters (p,d,q)(P,D,Q,s) are chosen as a robust default
    # for daily data with weekly seasonality (s=7).
    # order=(1,1,1) -> (p,d,q) for non-seasonal components
    # seasonal_order=(1,1,1,7) -> (P,D,Q,s) for weekly seasonality
    try:
        model = sm.tsa.statespace.SARIMAX(
            daily_counts,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Generate the forecast
        forecast = results.get_forecast(steps=forecast_days)
        
        # Get forecast values and confidence intervals
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Combine into a clean DataFrame
        forecast_df = pd.DataFrame({
            'forecast': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        })
        
        return daily_counts.reset_index(), forecast_df.reset_index()

    except Exception as e:
        # If the model fails to converge for any reason, return None
        print(f"Error during forecasting for cluster {cluster_id}: {e}")
        return None, None

