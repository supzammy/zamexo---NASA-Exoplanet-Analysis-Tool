import streamlit as st
import os
import sys

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

# Import path setup
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.title("⚙️ Settings & Configuration")

st.write("""
Configure the exoplanet analysis pipeline settings and preferences.
""")

# Analysis Settings
st.subheader("🔧 Analysis Settings")

with st.expander("🌌 Data Source Preferences", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Default NASA Mission Priority:**")
        mission_priority = st.multiselect(
            "Mission search order",
            ["TESS", "Kepler", "K2"],
            default=["TESS", "Kepler", "K2"],
            help="Order in which to search for light curve data"
        )
        
        default_sector_limit = st.number_input(
            "Default sector/quarter limit",
            min_value=1, max_value=50, value=5,
            help="Maximum number of sectors/quarters to download per target"
        )
    
    with col2:
        st.write("**Data Quality Filters:**")
        quality_bitmask = st.selectbox(
            "Quality bitmask",
            ["default", "hard", "hardest"],
            index=0,
            help="Strictness of data quality filtering"
        )
        
        cadence_preference = st.selectbox(
            "Cadence preference",
            ["long", "short", "fast"],
            index=0,
            help="Preferred observation cadence"
        )

# Save settings to session state
if st.button("💾 Save Data Settings"):
    st.session_state['mission_priority'] = mission_priority
    st.session_state['sector_limit'] = default_sector_limit
    st.session_state['quality_bitmask'] = quality_bitmask
    st.session_state['cadence_preference'] = cadence_preference
    st.success("✅ Data source settings saved!")

# Model Settings
st.subheader("🤖 Model Configuration")

with st.expander("🧠 Machine Learning Parameters", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Behavior:**")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1, max_value=0.9, value=0.7, step=0.05,
            help="Minimum confidence for positive predictions"
        )
        
        use_shap = st.checkbox(
            "Enable SHAP explanations",
            value=True,
            help="Generate explainable AI features (slower but more informative)"
        )
    
    with col2:
        st.write("**Feature Engineering:**")
        max_period_search = st.number_input(
            "Maximum period search [days]",
            min_value=1.0, max_value=100.0, value=20.0, step=1.0,
            help="Upper limit for BLS period search"
        )
        
        enable_detrending = st.checkbox(
            "Enable auto-detrending",
            value=True,
            help="Automatically remove long-term trends"
        )

# BLS Settings
with st.expander("📊 Transit Detection (BLS) Parameters"):
    col1, col2 = st.columns(2)
    
    with col1:
        duration_grid_factor = st.number_input(
            "Duration grid factor",
            min_value=0.5, max_value=3.0, value=1.0, step=0.1,
            help="Factor to multiply typical transit duration range"
        )
        
        oversample_factor = st.number_input(
            "Frequency oversampling",
            min_value=3, max_value=20, value=5,
            help="Factor to oversample frequency grid"
        )
    
    with col2:
        min_period = st.number_input(
            "Minimum period [days]",
            min_value=0.1, max_value=5.0, value=0.5, step=0.1,
            help="Lower limit for period search"
        )
        
        sde_threshold = st.number_input(
            "SDE threshold",
            min_value=3.0, max_value=10.0, value=5.0, step=0.5,
            help="Signal Detection Efficiency threshold"
        )

# Save model settings
if st.button("💾 Save Model Settings"):
    st.session_state.update({
        'confidence_threshold': confidence_threshold,
        'use_shap': use_shap,
        'max_period_search': max_period_search,
        'enable_detrending': enable_detrending,
        'duration_grid_factor': duration_grid_factor,
        'oversample_factor': oversample_factor,
        'min_period': min_period,
        'sde_threshold': sde_threshold,
    })
    st.success("✅ Model settings saved!")

# UI/UX Settings
st.subheader("🎨 Interface Preferences")

with st.expander("🖥️ Display Options", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Plot Settings:**")
        plot_theme = st.selectbox(
            "Plot theme",
            ["streamlit", "seaborn", "ggplot", "dark_background"],
            index=0
        )
        
        plot_dpi = st.selectbox(
            "Plot resolution",
            [100, 150, 200, 300],
            index=1,
            help="Higher DPI = sharper plots but larger files"
        )
        
        show_advanced_options = st.checkbox(
            "Show advanced options by default",
            value=False,
            help="Expand advanced parameter sections automatically"
        )
    
    with col2:
        st.write("**Data Display:**")
        max_plot_points = st.number_input(
            "Max points in plots",
            min_value=1000, max_value=50000, value=10000, step=1000,
            help="Subsample large datasets for faster plotting"
        )
        
        decimal_places = st.selectbox(
            "Metric decimal places",
            [2, 3, 4, 5],
            index=1,
            help="Precision for displayed numerical values"
        )
        
        enable_progress_bars = st.checkbox(
            "Show progress indicators",
            value=True,
            help="Display progress bars for long operations"
        )

# Save UI settings
if st.button("💾 Save Interface Settings"):
    st.session_state.update({
        'plot_theme': plot_theme,
        'plot_dpi': plot_dpi,
        'show_advanced_options': show_advanced_options,
        'max_plot_points': max_plot_points,
        'decimal_places': decimal_places,
        'enable_progress_bars': enable_progress_bars,
    })
    st.success("✅ Interface settings saved!")

# System Information
st.subheader("ℹ️ System Information")

with st.expander("📋 Environment Details"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Python Environment:**")
        st.code(f"Python: {sys.version}")
        st.code(f"Platform: {sys.platform}")
        
        # Try to get package versions
        try:
            import streamlit
            st.code(f"Streamlit: {streamlit.__version__}")
        except:
            st.code("Streamlit: unknown")
        
        try:
            import sklearn
            st.code(f"scikit-learn: {sklearn.__version__}")
        except:
            st.code("scikit-learn: unknown")
    
    with col2:
        st.write("**Key Libraries:**")
        
        libraries = [
            ('lightkurve', 'lightkurve'),
            ('astropy', 'astropy'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('matplotlib', 'matplotlib'),
            ('shap', 'shap')
        ]
        
        for display_name, import_name in libraries:
            try:
                lib = __import__(import_name)
                version = getattr(lib, '__version__', 'unknown')
                st.code(f"{display_name}: {version}")
            except ImportError:
                st.code(f"{display_name}: not installed")

# Current Settings Summary
st.subheader("📄 Current Settings Summary")

if st.button("🔄 Show Current Configuration"):
    st.write("**Active Settings:**")
    
    # Display current session state settings
    settings_to_show = [
        'mission_priority', 'sector_limit', 'quality_bitmask', 'cadence_preference',
        'confidence_threshold', 'use_shap', 'max_period_search', 'enable_detrending',
        'plot_theme', 'plot_dpi', 'max_plot_points', 'decimal_places'
    ]
    
    current_settings = {}
    for setting in settings_to_show:
        if setting in st.session_state:
            current_settings[setting] = st.session_state[setting]
    
    if current_settings:
        st.json(current_settings)
    else:
        st.info("No custom settings configured yet. Using defaults.")

# Reset Settings
st.subheader("🔄 Reset Options")

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Session Data", help="Clear all cached data and analysis results"):
        # Clear analysis data but keep settings
        keys_to_clear = [
            'light_curve_data', 'bls_results', 'ml_prediction', 
            'selected_target', 'synthetic_data'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Session data cleared!")

with col2:
    if st.button("⚙️ Reset All Settings", help="Reset all settings to defaults"):
        # Clear all custom settings
        settings_keys = [
            'mission_priority', 'sector_limit', 'quality_bitmask', 'cadence_preference',
            'confidence_threshold', 'use_shap', 'max_period_search', 'enable_detrending',
            'duration_grid_factor', 'oversample_factor', 'min_period', 'sde_threshold',
            'plot_theme', 'plot_dpi', 'show_advanced_options', 'max_plot_points', 
            'decimal_places', 'enable_progress_bars'
        ]
        for key in settings_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ All settings reset to defaults!")
        st.rerun()

# Help Section
st.subheader("❓ Help & Tips")

with st.expander("💡 Configuration Tips"):
    st.write("""
    **Performance Tips:**
    - Reduce 'Max points in plots' for faster rendering on large datasets
    - Disable SHAP explanations for faster predictions (less interpretable)
    - Use 'hard' quality bitmask for cleaner but smaller datasets
    
    **Analysis Tips:**
    - Increase 'Maximum period search' for longer-period planets (slower)
    - Lower 'Confidence threshold' to see more potential candidates
    - Adjust 'SDE threshold' based on noise level in your data
    
    **Troubleshooting:**
    - Reset session data if analysis results seem stale
    - Check system information if package-related errors occur
    - Contact support if persistent issues occur
    """)

st.markdown("---")
st.caption("⚙️ Settings are stored for this session only and will reset when you refresh the page.")