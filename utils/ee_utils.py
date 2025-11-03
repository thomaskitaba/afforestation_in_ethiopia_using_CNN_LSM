# utils/ee_utils.py
import streamlit as st
import pandas as pd
import numpy as np

def initialize_earth_engine():
    """Mock Earth Engine initialization for Streamlit Cloud"""
    st.success("🌱 Forest Growth Monitor - Demo Mode")
    st.info("📊 Using sample data for demonstration")
    return True

def create_aoi_geometry(min_lon, max_lon, min_lat, max_lat):
    """Create mock geometry"""
    return {"type": "rectangle", "coords": [min_lon, min_lat, max_lon, max_lat]}

def get_forest_data(geometry):
    """Get realistic sample forest data for Ethiopia"""
    try:
        # Realistic Ethiopian forest growth data (2018-2024)
        years = list(range(2018, 2025))
        data = []
        
        # Simulate successful reforestation progress
        ndvi_trend = [0.45, 0.52, 0.58, 0.63, 0.68, 0.72, 0.75]
        evi_trend = [0.35, 0.42, 0.48, 0.53, 0.57, 0.61, 0.64]
        cover_trend = [38, 45, 52, 58, 63, 68, 72]
        
        for i, year in enumerate(years):
            data.append({
                'year': year,
                'ndvi': round(ndvi_trend[i] + np.random.normal(0, 0.01), 3),
                'evi': round(evi_trend[i] + np.random.normal(0, 0.01), 3),
                'forest_cover': cover_trend[i]
            })
        
        return data
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return None

def get_vegetation_map(geometry, year):
    """Return a placeholder vegetation map image for demo purposes"""
    st.info(f"🗺️ Interactive vegetation map for {year}")
    # Return a static world map for demonstration
    return "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74192/world.topo.bathy.200412.3x5400x2700.jpg"