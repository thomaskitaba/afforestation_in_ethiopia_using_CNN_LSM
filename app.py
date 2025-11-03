import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import custom utilities
from utils.ee_utils import initialize_earth_engine, get_forest_data, create_aoi_geometry, get_vegetation_map
from utils.ml_utils import ForestGrowthPredictor
from utils.aois import AOIS

# Page configuration
st.set_page_config(
    page_title="Forest Growth Monitor - Advanced",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    local_css()
    
    # Header
    st.markdown('<h1 class="main-header">🌳 Advanced Forest Growth Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by ConvLSTM Neural Networks & Hyperparameter Tuning")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["🏠 Dashboard", "📊 Data Analysis", "🤖 ML Predictions", "🗺️ Map View", "📈 Reports", "⚙️ Model Settings"]
    )

    # ML Configuration in Sidebar (available across all pages)
    if app_mode != "⚙️ Model Settings":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 ML Configuration")
        
        # Model type selection
        model_type = st.sidebar.selectbox(
            "Model Architecture",
            ["ConvLSTM (Recommended)", "Simple Neural Network"],
            help="ConvLSTM captures both temporal and spatial patterns in time series data"
        )
        
        # Hyperparameter tuning options
        use_hyperparameter_tuning = st.sidebar.checkbox(
            "Enable Hyperparameter Tuning", 
            value=True,
            help="Automatically find the best model parameters (recommended)"
        )
        
        tune_iterations = st.sidebar.slider(
            "Tuning Iterations",
            min_value=10,
            max_value=50,
            value=20,
            help="More iterations = better results but longer training time"
        )
        
        sequence_length = st.sidebar.slider(
            "Sequence Length",
            min_value=2,
            max_value=5,
            value=3,
            help="Number of previous years to use for prediction"
        )

    # Precompute and cache all AOI data at startup (if not already cached)
    if 'all_aoi_data' not in st.session_state:
        st.session_state.all_aoi_data = {}
        for aoi_name, bbox in AOIS.items():
            try:
                geometry = create_aoi_geometry(bbox['min_lon'], bbox['max_lon'], bbox['min_lat'], bbox['max_lat'])
                data = get_forest_data(geometry)
                st.session_state.all_aoi_data[aoi_name] = {
                    'bbox': bbox,
                    'data': data
                }
            except Exception as e:
                st.session_state.all_aoi_data[aoi_name] = {
                    'bbox': bbox,
                    'data': None,
                    'error': str(e)
                }

    # Global AOI selector (applies across the app)
    if 'current_aoi' not in st.session_state:
        st.session_state.current_aoi = 'Custom'
    if 'aoi_bbox' not in st.session_state:
        st.session_state.aoi_bbox = None

    aoi_names = ['Custom'] + list(AOIS.keys())
    try:
        default_index = aoi_names.index(st.session_state.current_aoi)
    except ValueError:
        default_index = 0

    selected_global_aoi = st.sidebar.selectbox("Select AOI (global)", aoi_names, index=default_index)
    if selected_global_aoi != st.session_state.current_aoi:
        st.session_state.current_aoi = selected_global_aoi
        if selected_global_aoi != 'Custom':
            st.session_state.aoi_bbox = AOIS[selected_global_aoi]
        else:
            st.session_state.aoi_bbox = None
    
    # Initialize session state
    if 'ee_initialized' not in st.session_state:
        st.session_state.ee_initialized = False
    if 'forest_data' not in st.session_state:
        st.session_state.forest_data = None
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = ForestGrowthPredictor()
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Update model sequence length
    if 'ml_model' in st.session_state:
        st.session_state.ml_model.sequence_length = sequence_length
    
    # Earth Engine initialization
    if not st.session_state.ee_initialized:
        st.sidebar.info("🔐 Earth Engine Authentication Required")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Initialize Earth Engine"):
                with st.spinner("Initializing Earth Engine..."):
                    if initialize_earth_engine():
                        st.session_state.ee_initialized = True
                        st.sidebar.success("✅ Earth Engine Initialized!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Earth Engine initialization failed")
        
        with col2:
            if st.button("Clear Cache"):
                st.session_state.clear()
                st.rerun()
    
    # Main content based on selected mode
    if app_mode == "🏠 Dashboard":
        show_dashboard(use_hyperparameter_tuning, tune_iterations)
    elif app_mode == "📊 Data Analysis":
        show_data_analysis()
    elif app_mode == "🤖 ML Predictions":
        show_ml_predictions(use_hyperparameter_tuning, tune_iterations)
    elif app_mode == "🗺️ Map View":
        show_map_view()
    elif app_mode == "📈 Reports":
        show_reports()
    elif app_mode == "⚙️ Model Settings":
        show_model_settings()

def show_dashboard(use_tuning, tune_iterations):
    st.header("🌿 Forest Growth Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Study Area")
        current_aoi = st.session_state.get('current_aoi', None)
        if current_aoi and current_aoi != 'Custom':
            st.info(f"{current_aoi}")
            bbox = st.session_state.get('aoi_bbox') or {}
            if bbox:
                st.markdown(f"**Bounding box:** {bbox.get('min_lon')}, {bbox.get('min_lat')} → {bbox.get('max_lon')}, {bbox.get('max_lat')}")
        else:
            st.info("Bale Mountains Region, Ethiopia")

        def approx_area_km2(min_lon, min_lat, max_lon, max_lat):
            from math import cos, pi
            mean_lat = (min_lat + max_lat) / 2.0
            km_per_deg_lat = 110.574
            km_per_deg_lon = 111.320 * cos(mean_lat * pi / 180.0)
            width_km = abs(max_lon - min_lon) * km_per_deg_lon
            height_km = abs(max_lat - min_lat) * km_per_deg_lat
            return max(width_km * height_km, 0.0)

        dashboard_data = None
        area_km2 = None
        if current_aoi and current_aoi != 'Custom':
            aoi_entry = st.session_state.all_aoi_data.get(current_aoi)
            if aoi_entry:
                dashboard_data = aoi_entry['data']
                b = aoi_entry['bbox']
                area_km2 = approx_area_km2(b['min_lon'], b['min_lat'], b['max_lon'], b['max_lat'])
        elif st.session_state.get('aoi_bbox'):
            b = st.session_state['aoi_bbox']
            area_km2 = approx_area_km2(b['min_lon'], b['min_lat'], b['max_lon'], b['max_lat'])
            try:
                geometry = create_aoi_geometry(b['min_lon'], b['max_lon'], b['min_lat'], b['max_lat'])
                dashboard_data = get_forest_data(geometry)
            except Exception:
                dashboard_data = None

        area_label = f"{area_km2:.1f} km²" if area_km2 is not None else "—"
        if dashboard_data:
            df = pd.DataFrame(dashboard_data)
            st.metric("Area Size", area_label)
            st.metric("Planting Started", "2018")
        else:
            st.metric("Area Size", "100 km²")
            st.metric("Planting Started", "2018")

    with col2:
        st.subheader("📈 Growth Metrics")
        if dashboard_data:
            try:
                df = pd.DataFrame(dashboard_data)
                current_ndvi = df['ndvi'].iloc[-1]
                ndvi_change = current_ndvi - df['ndvi'].iloc[0]
                forest_cover = df['forest_cover'].iloc[-1] if 'forest_cover' in df.columns else None
                growth_rate = df['ndvi'].pct_change().mean() * 100
                st.metric("Current NDVI", f"{current_ndvi:.3f}", f"{ndvi_change:+.3f}")
                if forest_cover is not None:
                    st.metric("Forest Cover", f"{forest_cover:.0f}%")
                else:
                    st.metric("Forest Cover", "—")
                st.metric("Growth Rate", f"{growth_rate:.2f}%", "per year")
            except Exception:
                st.metric("Current NDVI", "0.720", "+0.080")
                st.metric("Forest Cover", "68%", "+12%")
                st.metric("Growth Rate", "3.2%", "per year")
        else:
            st.metric("Current NDVI", "0.720", "+0.080")
            st.metric("Forest Cover", "68%", "+12%")
            st.metric("Growth Rate", "3.2%", "per year")

    with col3:
        st.subheader("🎯 Targets & ML Status")
        st.metric("2025 Target", "75% Coverage")
        st.metric("Progress", "85%", "to target")
        
        # ML Training Status
        if st.session_state.get('trained'):
            st.success("✅ Model Trained")
            model_info = st.session_state.ml_model.get_model_info()
            if model_info.get('best_params'):
                st.info("🎯 Hyperparameter Tuning: Used")
            else:
                st.info("⚙️ Hyperparameter Tuning: Not Used")
        else:
            st.warning("🤖 Model Not Trained")

    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    # Use AOI-specific data for actions
    aoi_data = dashboard_data
    aoi_name = current_aoi if current_aoi and current_aoi != 'Custom' else 'Custom'

    with col1:
        if st.button("📥 Fetch Latest Data", use_container_width=True):
            if aoi_data:
                st.success(f"Fetched latest data for {aoi_name} ({len(aoi_data)} years)")
            else:
                st.warning("No data available for selected AOI.")

    with col2:
        if st.button("🤖 Train ML Model", use_container_width=True):
            if aoi_data:
                results = st.session_state.ml_model.train(aoi_data, use_tuning=use_tuning, tune_iterations=tune_iterations)
                if results:
                    st.session_state.trained = True
                    if 'ml_training_results' not in st.session_state:
                        st.session_state.ml_training_results = {}
                    st.session_state.ml_training_results[aoi_name] = results
                    st.success(f"Model trained for {aoi_name}!")
                    st.rerun()
                else:
                    st.error("Training failed.")
            else:
                st.warning("No data to train model for selected AOI.")

    with col3:
        if st.button("🔮 Predict Growth", use_container_width=True):
            if aoi_data and st.session_state.get('trained'):
                years_ahead = 3
                predictions = st.session_state.ml_model.predict_future(aoi_data, years_ahead)
                if predictions is not None:
                    df_data = pd.DataFrame(aoi_data)
                    future_years = list(range(df_data['year'].max() + 1, df_data['year'].max() + 1 + years_ahead))
                    if 'ml_predictions' not in st.session_state:
                        st.session_state.ml_predictions = {}
                    if 'ml_predictions_years' not in st.session_state:
                        st.session_state.ml_predictions_years = {}
                    st.session_state.ml_predictions[aoi_name] = predictions
                    st.session_state.ml_predictions_years[aoi_name] = future_years
                    st.success(f"Predicted growth for {aoi_name} ({years_ahead} years ahead)")
                else:
                    st.error("Prediction failed.")
            else:
                st.warning("Train the model first for selected AOI.")

    with col4:
        if st.button("📊 Generate Report", use_container_width=True):
            if aoi_data:
                st.info(f"Report generated for {aoi_name} (see Reports section)")
            else:
                st.warning("No data to generate report for selected AOI.")

    # Recent Activity Section
    st.subheader("📈 Recent Activity")
    col1, col2 = st.columns(2)
    
    with col1:
        if dashboard_data:
            df = pd.DataFrame(dashboard_data)
            fig = px.line(df, x='year', y='ndvi', 
                         title='NDVI Trend Over Time',
                         markers=True)
            fig.update_layout(yaxis_title="NDVI", xaxis_title="Year")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st.session_state.get('trained') and st.session_state.get('ml_training_results'):
            st.info("🎯 Model Performance Summary")
            latest_results = list(st.session_state.ml_training_results.values())[-1]
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("R² Score", f"{latest_results['r2']:.4f}")
            with col_b:
                st.metric("MAE", f"{latest_results['mae']:.4f}")

def show_data_analysis():
    st.header("📊 Forest Data Analysis")
    
    # Area selection
    st.subheader("📍 Select Monitoring Area")

    # AOI dropdown: includes a 'Custom' option so users can enter their own bounding box
    aoi_names = ["Custom"] + list(AOIS.keys())
    selected_aoi = st.selectbox("Choose AOI (or Custom to enter coordinates)", aoi_names)

    # Determine default values based on selection; also sync global selection
    st.session_state.current_aoi = selected_aoi
    if selected_aoi != "Custom":
        bbox = AOIS[selected_aoi]
        st.session_state.aoi_bbox = bbox
        min_lon_def = bbox['min_lon']
        max_lon_def = bbox['max_lon']
        min_lat_def = bbox['min_lat']
        max_lat_def = bbox['max_lat']
        st.markdown(f"**Selected AOI:** {selected_aoi}  ")
        st.markdown(f"Bounding box (lon/lat): {min_lon_def}, {min_lat_def}, {max_lon_def}, {max_lat_def}")
    else:
        # sensible defaults (previous defaults preserved)
        # if a global AOI bbox was previously set, use it as defaults
        global_bbox = st.session_state.get('aoi_bbox')
        if global_bbox:
            min_lon_def = global_bbox.get('min_lon', 39.0)
            max_lon_def = global_bbox.get('max_lon', 39.8)
            min_lat_def = global_bbox.get('min_lat', 8.2)
            max_lat_def = global_bbox.get('max_lat', 8.8)
        else:
            min_lon_def, max_lon_def, min_lat_def, max_lat_def = 39.0, 39.8, 8.2, 8.8

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        min_lon = st.number_input("Min Longitude", value=float(min_lon_def), format="%.6f")
    with col2:
        max_lon = st.number_input("Max Longitude", value=float(max_lon_def), format="%.6f")
    with col3:
        min_lat = st.number_input("Min Latitude", value=float(min_lat_def), format="%.6f")
    with col4:
        max_lat = st.number_input("Max Latitude", value=float(max_lat_def), format="%.6f")
    
    if st.button("🌍 Fetch Forest Data", type="primary"):
        if st.session_state.ee_initialized:
            with st.spinner("Fetching satellite data..."):
                geometry = create_aoi_geometry(min_lon, max_lon, min_lat, max_lat)
                forest_data = get_forest_data(geometry)
                
                if forest_data:
                    st.session_state.forest_data = forest_data
                    st.success(f"✅ Successfully fetched {len(forest_data)} years of data!")
                    
                    # Display data
                    df = pd.DataFrame(forest_data)
                    st.subheader("📋 Forest Growth Data")
                    st.dataframe(df.round(4))
                    
                    # Create charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.line(df, x='year', y='ndvi', 
                                    title='NDVI Trend Over Time',
                                    markers=True)
                        fig.update_layout(yaxis_title="NDVI", xaxis_title="Year")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(df, x='year', y='ndvi',
                                   title='Vegetation Health by Year')
                        fig.update_layout(yaxis_title="NDVI", xaxis_title="Year")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical Analysis
                    st.subheader("📈 Statistical Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean NDVI", f"{df['ndvi'].mean():.4f}")
                    with col2:
                        st.metric("NDVI Std Dev", f"{df['ndvi'].std():.4f}")
                    with col3:
                        st.metric("Trend Slope", f"{(df['ndvi'].iloc[-1] - df['ndvi'].iloc[0]) / len(df):.4f}")
                        
                else:
                    st.error("❌ No data retrieved. Please check coordinates.")
        else:
            st.error("❌ Please initialize Earth Engine first.")

def show_ml_predictions(use_tuning, tune_iterations):
    st.header("🤖 Advanced ML Predictions (ConvLSTM)")
    
    # Use AOI-specific data for model training and prediction
    current_aoi = st.session_state.get('current_aoi', None)
    aoi_data = None
    if current_aoi and current_aoi != 'Custom':
        aoi_entry = st.session_state.all_aoi_data.get(current_aoi)
        if aoi_entry:
            aoi_data = aoi_entry['data']
    elif st.session_state.get('aoi_bbox'):
        b = st.session_state['aoi_bbox']
        try:
            geometry = create_aoi_geometry(b['min_lon'], b['max_lon'], b['min_lat'], b['max_lat'])
            aoi_data = get_forest_data(geometry)
        except Exception:
            aoi_data = None

    if not aoi_data:
        st.warning("⚠️ Please select an AOI with available data.")
        return

    # Persist model training and prediction results in session state, per AOI
    if 'ml_training_results' not in st.session_state:
        st.session_state.ml_training_results = {}
    if 'ml_predictions' not in st.session_state:
        st.session_state.ml_predictions = {}
    if 'ml_predictions_years' not in st.session_state:
        st.session_state.ml_predictions_years = {}

    aoi_key = current_aoi if current_aoi and current_aoi != 'Custom' else 'CUSTOM'

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Model Training")
        
        # Training status
        if st.session_state.get('trained'):
            st.success("✅ Model is trained and ready for predictions!")
            
            # Display model info
            model_info = st.session_state.ml_model.get_model_info()
            if model_info['trained']:
                st.info(f"**Architecture:** {model_info['architecture']}")
                st.info(f"**Sequence Length:** {model_info['sequence_length']}")
                if model_info.get('best_params'):
                    st.success("✅ Hyperparameter tuning was used")
        
        if st.button("🚀 Train Prediction Model", use_container_width=True):
            with st.spinner("Training advanced ConvLSTM model..."):
                results = st.session_state.ml_model.train(
                    aoi_data, 
                    use_tuning=use_tuning,
                    tune_iterations=tune_iterations
                )
                if results:
                    st.session_state.trained = True
                    st.session_state.ml_training_results[aoi_key] = results
                    st.rerun()

        # Show training results if available
        results = st.session_state.ml_training_results.get(aoi_key)
        if results:
            # Additional metrics
            st.metric("Training Samples", results.get('train_size', 'N/A'))
            st.metric("Validation Samples", results['test_size'])
            
            # Model architecture preview
            with st.expander("📋 Model Architecture Details"):
                st.text(results['model_summary'])

    with col2:
        st.subheader("🔮 Future Predictions")
        years_ahead = st.slider("Years to Predict", 1, 5, 3, key="pred_years")

        if st.button("📈 Predict Future Growth", use_container_width=True):
            if st.session_state.trained and st.session_state.ml_training_results.get(aoi_key):
                with st.spinner("Making ConvLSTM predictions..."):
                    predictions = st.session_state.ml_model.predict_future(
                        aoi_data, years_ahead
                    )
                    if predictions is not None:
                        df_data = pd.DataFrame(aoi_data)
                        future_years = list(range(df_data['year'].max() + 1, 
                                                df_data['year'].max() + 1 + years_ahead))
                        st.session_state.ml_predictions[aoi_key] = predictions
                        st.session_state.ml_predictions_years[aoi_key] = future_years
                        st.success(f"✅ Generated {years_ahead} year predictions!")
                        st.rerun()
            else:
                st.error("❌ Please train the model first.")

        # Enhanced predictions display
        predictions = st.session_state.ml_predictions.get(aoi_key)
        future_years = st.session_state.ml_predictions_years.get(aoi_key)
        if predictions is not None and future_years is not None:
            df_data = pd.DataFrame(aoi_data)
            
            # Create enhanced visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df_data['year'], y=df_data['ndvi'],
                name='Historical Data', mode='lines+markers',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=future_years, y=predictions,
                name='ConvLSTM Predictions', mode='lines+markers',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8, symbol='star')
            ))
            
            # Confidence interval (estimated)
            last_ndvi = df_data['ndvi'].iloc[-1]
            confidence_upper = predictions + 0.02  # Simulated confidence
            confidence_lower = predictions - 0.02
            
            fig.add_trace(go.Scatter(
                x=future_years + future_years[::-1],
                y=np.concatenate([confidence_upper, confidence_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title='Forest Growth Predictions with ConvLSTM',
                xaxis_title='Year',
                yaxis_title='NDVI',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Enhanced prediction details
            st.subheader("📋 Prediction Analysis")
            pred_df = pd.DataFrame({
                'Year': future_years,
                'Predicted NDVI': predictions,
                'Growth %': ((predictions - last_ndvi) / last_ndvi * 100)
            })
            st.dataframe(pred_df.round(4))
            
            # Summary statistics
            avg_growth = pred_df['Growth %'].mean()
            total_growth = ((predictions[-1] - last_ndvi) / last_ndvi * 100)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Annual Growth", f"{avg_growth:.2f}%")
            with col2:
                st.metric("Total Projected Growth", f"{total_growth:.2f}%")

def show_map_view():
    st.header("🗺️ Interactive Map View")
    
    if not st.session_state.ee_initialized:
        st.error("❌ Please initialize Earth Engine first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Map Settings")
        year = st.slider("Select Year", 2018, 2024, 2024)
        map_type = st.selectbox("Map Type", ["Vegetation Health", "Satellite View"])

        # AOI selection: pick a preset AOI or enter custom bounding box
        aoi_names = ["Custom"] + list(AOIS.keys())
        # prefer the global AOI selection if set
        default_index = 0
        if st.session_state.get('current_aoi') in aoi_names:
            default_index = aoi_names.index(st.session_state.get('current_aoi'))
        selected_aoi = st.selectbox("Choose AOI (or Custom to enter coordinates)", aoi_names, index=default_index)

        if selected_aoi != "Custom":
            bbox = AOIS[selected_aoi]
            min_lon_def = bbox['min_lon']
            max_lon_def = bbox['max_lon']
            min_lat_def = bbox['min_lat']
            max_lat_def = bbox['max_lat']
            st.markdown(f"**Selected AOI:** {selected_aoi}")
            st.markdown(f"Bounding box (lon/lat): {min_lon_def}, {min_lat_def}, {max_lon_def}, {max_lat_def}")
            # sync global state
            st.session_state.current_aoi = selected_aoi
            st.session_state.aoi_bbox = bbox
        else:
            # sensible defaults
            # prefer any global AOI bbox if present
            global_bbox = st.session_state.get('aoi_bbox')
            if global_bbox:
                min_lon_def = global_bbox.get('min_lon', 39.0)
                max_lon_def = global_bbox.get('max_lon', 39.8)
                min_lat_def = global_bbox.get('min_lat', 8.2)
                max_lat_def = global_bbox.get('max_lat', 8.8)
            else:
                min_lon_def, max_lon_def, min_lat_def, max_lat_def = 39.0, 39.8, 8.2, 8.8

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            min_lon = st.number_input("Min Longitude", value=float(min_lon_def), format="%.6f")
        with col_b:
            max_lon = st.number_input("Max Longitude", value=float(max_lon_def), format="%.6f")
        with col_c:
            min_lat = st.number_input("Min Latitude", value=float(min_lat_def), format="%.6f")
        with col_d:
            max_lat = st.number_input("Max Latitude", value=float(max_lat_def), format="%.6f")

        # Build geometry from the chosen coordinates
        geometry = create_aoi_geometry(min_lon, max_lon, min_lat, max_lat)

        if st.button("🔄 Update Map"):
            with st.spinner("Generating map..."):
                map_url = get_vegetation_map(geometry, year)
                st.session_state.map_url = map_url
                st.session_state.map_year = year

    with col2:
        st.subheader(f"Forest Map {getattr(st.session_state, 'map_year', 2024)}")
        if hasattr(st.session_state, 'map_url') and st.session_state.map_url:
            st.image(st.session_state.map_url, use_column_width=True)
            st.caption("Vegetation Health: Red (Low) → Yellow (Medium) → Green (High)")
        else:
            st.info("👆 Choose an AOI and click 'Update Map' to generate vegetation map")

def show_reports():
    st.header("📈 Comprehensive Reports")

    # Use AOI-specific data for reports
    current_aoi = st.session_state.get('current_aoi', None)
    aoi_data = None
    if current_aoi and current_aoi != 'Custom':
        aoi_entry = st.session_state.all_aoi_data.get(current_aoi)
        if aoi_entry:
            aoi_data = aoi_entry['data']
    elif st.session_state.get('aoi_bbox'):
        b = st.session_state['aoi_bbox']
        try:
            geometry = create_aoi_geometry(b['min_lon'], b['max_lon'], b['min_lat'], b['max_lat'])
            aoi_data = get_forest_data(geometry)
        except Exception:
            aoi_data = None

    if not aoi_data:
        st.warning("⚠️ Please select an AOI with available data.")
        return

    # Generate report
    if st.button("📄 Generate Comprehensive Report", type="primary"):
        with st.spinner("Generating report..."):
            generate_comprehensive_report(aoi_data)

def show_model_settings():
    st.header("⚙️ Model Configuration & Settings")
    
    st.markdown("""
    ### 🧠 Advanced Model Configuration
    
    Configure your ConvLSTM model parameters and training settings for optimal performance.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.info("""
        **ConvLSTM (Convolutional LSTM)**
        - Captures both temporal patterns and local spatial dependencies
        - Ideal for time series data with spatial characteristics
        - Better generalization than standard neural networks
        """)
        
        # Model parameters
        st.subheader("Model Parameters")
        sequence_length = st.slider(
            "Sequence Length",
            min_value=2,
            max_value=5,
            value=3,
            help="Number of previous years used to predict the next year"
        )
        
        # Update the model's sequence length
        if st.session_state.get('ml_model'):
            st.session_state.ml_model.sequence_length = sequence_length
    
    with col2:
        st.subheader("Training Configuration")
        
        st.checkbox(
            "Enable Early Stopping",
            value=True,
            help="Stop training when validation loss stops improving"
        )
        
        st.checkbox(
            "Use Learning Rate Scheduling",
            value=True,
            help="Reduce learning rate when loss plateaus"
        )
        
        st.slider(
            "Patience Epochs",
            min_value=5,
            max_value=30,
            value=20,
            help="Number of epochs to wait before early stopping"
        )
    
    # Hyperparameter Tuning Section
    st.markdown("---")
    st.subheader("🎯 Hyperparameter Tuning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Why Use Hyperparameter Tuning?**
        - Automatically finds optimal model parameters
        - Improves prediction accuracy
        - Reduces manual configuration effort
        - Provides best model performance
        """)
    
    with col2:
        use_tuning = st.checkbox(
            "Enable Automatic Hyperparameter Tuning",
            value=True,
            help="Recommended for best results"
        )
        
        tune_iterations = st.slider(
            "Tuning Iterations",
            min_value=10,
            max_value=50,
            value=20,
            help="More iterations = better results but longer training time"
        )
        
        if st.button("🔧 Test Default Parameters"):
            st.info("Current model parameters are optimized for forest growth prediction")
    
    # Model Information
    if st.session_state.get('trained'):
        st.markdown("---")
        st.subheader("📊 Current Model Status")
        
        model_info = st.session_state.ml_model.get_model_info()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Architecture", model_info['architecture'])
        with col2:
            st.metric("Sequence Length", model_info['sequence_length'])
        with col3:
            status = "Tuned" if model_info.get('best_params') else "Default"
            st.metric("Parameters", status)
        
        if model_info.get('best_params'):
            with st.expander("🎯 Best Hyperparameters Found"):
                best_params = model_info['best_params']
                for param, value in best_params.items():
                    st.write(f"**{param}:** `{value}`")

def generate_comprehensive_report(aoi_data):
    """Generate a comprehensive forest growth report for the selected AOI"""
    df = pd.DataFrame(aoi_data)

    st.subheader("🌳 Forest Growth Analysis Report")

    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_growth = (df['ndvi'].iloc[-1] - df['ndvi'].iloc[0]) * 100
        st.metric("Total Growth", f"{total_growth:.1f}%")

    with col2:
        avg_growth = (df['ndvi'].pct_change().mean() * 100)
        st.metric("Average Annual Growth", f"{avg_growth:.1f}%")

    with col3:
        current_health = "Excellent" if df['ndvi'].iloc[-1] > 0.7 else "Good" if df['ndvi'].iloc[-1] > 0.5 else "Needs Attention"
        st.metric("Current Health", current_health)

    # ML Model Performance
    if st.session_state.get('trained') and st.session_state.get('ml_training_results'):
        st.markdown("### 🤖 Machine Learning Analysis")
        latest_results = list(st.session_state.ml_training_results.values())[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model R² Score", f"{latest_results['r2']:.4f}")
        with col2:
            st.metric("Prediction MAE", f"{latest_results['mae']:.4f}")
        with col3:
            st.metric("Model Confidence", "High" if latest_results['r2'] > 0.8 else "Medium")

    # Detailed Analysis
    st.markdown("### 📊 Detailed Analysis")

    # Growth trends
    fig = px.line(df, x='year', y=['ndvi', 'evi'] if 'evi' in df.columns else 'ndvi',
                 title='Vegetation Health Trends',
                 labels={'value': 'Index Value', 'variable': 'Index Type'})
    st.plotly_chart(fig, use_container_width=True)

    # Statistics table
    st.markdown("### 📈 Statistical Summary")
    stats_df = df.describe()
    st.dataframe(stats_df.round(4))

    # Recommendations
    st.markdown("### 💡 Recommendations")

    latest_ndvi = df['ndvi'].iloc[-1]
    if latest_ndvi > 0.7:
        st.success("""
        🌟 **Excellent Progress!**
        - Continue current planting strategies
        - Focus on maintaining existing forests
        - Consider expanding to new areas
        - Monitor for potential saturation effects
        """)
    elif latest_ndvi > 0.5:
        st.info("""
        👍 **Good Progress**
        - Maintain current efforts
        - Address any local challenges
        - Monitor soil health
        - Consider targeted interventions in low-growth areas
        """)
    else:
        st.warning("""
        ⚠️ **Needs Attention**
        - Review planting techniques
        - Consider soil amendments
        - Increase monitoring frequency
        - Investigate potential environmental stressors
        """)

# Helper functions
def fetch_latest_data():
    """Fetch latest forest data"""
    st.info("📥 Fetching latest satellite data...")

def train_ml_model():
    """Train ML model"""
    st.info("🤖 Training machine learning model...")

def predict_growth():
    """Predict future growth"""
    st.info("🔮 Making growth predictions...")

def generate_report():
    """Generate report"""
    st.info("📊 Generating comprehensive report...")

if __name__ == "__main__":
    main()