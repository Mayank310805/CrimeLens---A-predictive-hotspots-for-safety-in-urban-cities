# ml_module.py
# This file contains the functions for machine learning analysis.

from sklearn.cluster import DBSCAN, KMeans
import pandas as pd

def detect_hotspots_dbscan(data, eps=0.01, min_samples=5):
    """
    Detects crime hotspots using the DBSCAN algorithm.
    - 'eps' is the max distance between two samples for one to be considered as in the neighborhood of the other.
    - 'min_samples' is the number of samples in a neighborhood for a point to be considered as a core point.
    - Uses the haversine metric for geographic coordinates.
    """
    # Ensure data has latitude and longitude
    if 'latitude' not in data.columns or 'longitude' not in data.columns:
        raise ValueError("Input data must contain 'latitude' and 'longitude' columns.")
        
    coords = data[['latitude', 'longitude']].values
    
    # DBSCAN works with radians for haversine metric
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    
    # Add cluster labels to the dataframe. -1 indicates noise (not part of a cluster).
    data['cluster'] = db.labels_
    return data

def detect_hotspots_kmeans(data, n_clusters=5):
    """
    Detects crime hotspots using the K-Means algorithm.
    - 'n_clusters' is the number of clusters (hotspots) to find.
    """
    # Ensure data has latitude and longitude
    if 'latitude' not in data.columns or 'longitude' not in data.columns:
        raise ValueError("Input data must contain 'latitude' and 'longitude' columns.")

    coords = data[['latitude', 'longitude']].values
    
    # n_init='auto' is the modern default to avoid FutureWarnings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(coords)
    
    data['cluster'] = kmeans.labels_
    return data
