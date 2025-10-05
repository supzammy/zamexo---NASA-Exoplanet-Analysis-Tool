import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")

# Import path setup
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def _clean_series(x):
    if hasattr(x, 'value'):
        x = x.value
    arr = np.asarray(x, dtype=float)
    if np.ma.isMaskedArray(arr):
        arr = np.asarray(arr.filled(np.nan), dtype=float)
    return arr

def summarize_lightcurve(time, flux):
    t = _clean_series(time); f = _clean_series(flux)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size == 0:
        return {}
    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med)) * 1.4826 if np.isfinite(med) else np.nan
    dt = np.diff(np.sort(t))
    return {
        'n_points': int(f.size),
        'baseline_days': float(t.max() - t.min()) if t.size else 0,
        'cadence_days': float(np.median(dt)) if dt.size else np.nan,
        'mean_flux': float(np.nanmean(f)) if f.size else np.nan,
        'std_flux': float(np.nanstd(f)) if f.size else np.nan,
        'median_flux': float(med),
        'robust_rms': float(mad),
        'frac_rms': float((np.nanstd(f)/med) if (f.size and med!=0) else np.nan),
    }

def simple_bls(time, flux, max_period):
    t = _clean_series(time); f = _clean_series(flux)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 10:
        return {'period': np.nan,'duration': np.nan,'depth': np.nan,'sde': np.nan,'t0': np.nan}
    t = t - t.min()
    med = np.nanmedian(f)
    if np.isfinite(med) and med!=0:
        f = f/med - 1.0
    else:
        f = f - np.nanmean(f)
    n = min(f.size, 4000)
    f2 = f[:n] - np.nanmean(f[:n])
    try:
        ac = np.correlate(f2, f2, mode='full')[n-1:]
    except TypeError:
        return {'period': np.nan,'duration': np.nan,'depth': np.nan,'sde': np.nan,'t0': np.nan}
    ac[0]=0
    lag = int(np.argmax(ac[1:])+1)
    dt = np.median(np.diff(t)) if t.size>1 else np.nan
    period = float(lag*dt) if np.isfinite(dt) else np.nan
    if not (np.isfinite(period) and 0 < period <= max_period):
        period = np.nan
    depth = float(np.nanpercentile(f,2)) if np.isfinite(period) else np.nan
    return {'period': period,'duration': period*0.05 if np.isfinite(period) else np.nan,'depth': depth,'sde': np.nan,'t0': float(t.min())}

st.title("📤 Data Upload")

st.write("""
Upload your own light curve data for analysis! This page allows you to:
- 📊 **Upload CSV files** with time and flux measurements
- 🔍 **Preview and validate** your data 
- 🚀 **Run the full analysis pipeline** (Transit Detection → AI Classification → Explainability)
""")

# File upload section
st.subheader("📁 File Upload")

uploaded_file = st.file_uploader(
    "Choose a CSV file with light curve data",
    type=['csv'],
    help="CSV should contain columns named 'time' and 'flux', or the first two numeric columns will be used."
)

# Data format help
with st.expander("📋 Data Format Requirements"):
    st.write("""
    **Required Format:**
    - CSV file with time and flux measurements
    - Column names: `time`, `flux` (preferred) or first two numeric columns
    - Time: typically in days (any reference point)
    - Flux: normalized stellar brightness (e.g., 1.0 = baseline)
    
    **Example format:**
    ```csv
    time,flux
    0.0,1.001
    0.02,0.999
    0.04,1.002
    ...
    ```
    
    **Tips:**
    - Remove or mark bad data points as NaN
    - Ensure reasonable time sampling (not too sparse)
    - Flux should be normalized (around 1.0 for no transit)
    - At least 1000+ points recommended for reliable analysis
    """)

# Sample data generator
with st.expander("🧪 Generate Sample Data"):
    st.write("Create synthetic light curve data for testing:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.number_input("Period [days]", value=2.5, min_value=0.5, max_value=10.0)
    with col2:
        depth = st.number_input("Transit depth", value=0.01, min_value=0.001, max_value=0.1, format="%.3f")
    with col3:
        noise_level = st.number_input("Noise level", value=0.001, min_value=0.0001, max_value=0.01, format="%.4f")
    
    if st.button("📊 Generate Sample Data"):
        # Create synthetic light curve
        rng = np.random.default_rng(42)
        time_synth = np.linspace(0, 20, 4000)  # 20 days, 4000 points
        
        # Add transit signals
        flux_synth = np.ones_like(time_synth)
        transit_duration = period * 0.05  # 5% of period
        
        for cycle in range(int(20 / period) + 1):
            transit_center = cycle * period
            transit_mask = np.abs(time_synth - transit_center) < (transit_duration / 2)
            flux_synth[transit_mask] -= depth
        
        # Add noise
        flux_synth += rng.normal(0, noise_level, size=len(flux_synth))
        
        # Store in session state
        synthetic_df = pd.DataFrame({
            'time': time_synth,
            'flux': flux_synth
        })
        st.session_state['synthetic_data'] = synthetic_df
        st.success(f"✅ Generated synthetic data with P={period:.2f}d, depth={depth:.3f}")

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Successfully loaded {uploaded_file.name}")
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.write(f"**Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column detection
        if 'time' in df.columns and 'flux' in df.columns:
            time_col, flux_col = 'time', 'flux'
            st.success("✅ Found 'time' and 'flux' columns")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                time_col, flux_col = numeric_cols[0], numeric_cols[1]
                st.info(f"ℹ️ Using first two numeric columns: '{time_col}' (time), '{flux_col}' (flux)")
            else:
                st.error("❌ Need at least two numeric columns")
                st.stop()
        
        # Extract and validate data
        time_data = df[time_col].dropna()
        flux_data = df[flux_col].dropna()
        
        # Align arrays (remove NaN pairs)
        valid_mask = pd.notna(df[time_col]) & pd.notna(df[flux_col])
        time_clean = df.loc[valid_mask, time_col].values
        flux_clean = df.loc[valid_mask, flux_col].values
        
        if len(time_clean) < 100:
            st.error("❌ Need at least 100 valid data points")
            st.stop()
        
        # Data quality assessment
        st.subheader("📊 Data Quality Assessment")
        
        stats = summarize_lightcurve(time_clean, flux_clean)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valid Points", f"{stats['n_points']:,}")
        col2.metric("Baseline [d]", f"{stats['baseline_days']:.2f}")
        col3.metric("Cadence [min]", f"{stats['cadence_days']*1440:.1f}" if np.isfinite(stats['cadence_days']) else "–")
        col4.metric("RMS", f"{stats['frac_rms']:.3%}" if np.isfinite(stats['frac_rms']) else "–")
        
        # Quality indicators
        quality_issues = []
        if stats['n_points'] < 1000:
            quality_issues.append("⚠️ Low point count (< 1000)")
        if stats['baseline_days'] < 5:
            quality_issues.append("⚠️ Short baseline (< 5 days)")
        if stats['frac_rms'] > 0.05:
            quality_issues.append("⚠️ High noise level (> 5%)")
        
        if quality_issues:
            st.warning("**Data Quality Issues:**")
            for issue in quality_issues:
                st.write(f"- {issue}")
        else:
            st.success("✅ Data quality looks good!")
        
        # Plot the light curve
        st.subheader("📈 Light Curve Visualization")
        
        # Optionally subsample for plotting
        if len(time_clean) > 10000:
            plot_indices = np.linspace(0, len(time_clean)-1, 10000, dtype=int)
            time_plot = time_clean[plot_indices]
            flux_plot = flux_clean[plot_indices]
            st.caption("(Subsampled to 10k points for visualization)")
        else:
            time_plot = time_clean
            flux_plot = flux_clean
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_plot, flux_plot, '.', ms=1, alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.set_title(f'Light Curve: {uploaded_file.name}')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        
        # Analysis controls
        st.subheader("🔧 Analysis Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            max_period = st.slider("Maximum period to search [days]", 
                                 min_value=0.5, 
                                 max_value=min(50.0, stats['baseline_days']/3), 
                                 value=min(10.0, stats['baseline_days']/3),
                                 step=0.5)
        with col2:
            run_full_analysis = st.button("🚀 Run Full Analysis Pipeline", type="primary")
        
        if run_full_analysis:
            # Store data in session state for other pages
            st.session_state['light_curve_data'] = (time_clean, flux_clean)
            st.session_state['source_label'] = f"Upload: {uploaded_file.name}"
            
            # Run BLS analysis
            with st.spinner("Running transit detection..."):
                bls_results = simple_bls(time_clean, flux_clean, max_period)
            
            st.session_state['bls_results'] = bls_results
            
            # Show results
            st.success("✅ Analysis complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Period [d]", f"{bls_results['period']:.4g}" if np.isfinite(bls_results['period']) else "–")
            col2.metric("Duration [d]", f"{bls_results['duration']:.4g}" if np.isfinite(bls_results['duration']) else "–")
            depth_ppm = bls_results['depth']*1e6 if np.isfinite(bls_results.get('depth', np.nan)) else np.nan
            col3.metric("Depth [ppm]", f"{depth_ppm:.0f}" if np.isfinite(depth_ppm) else "–")
            
            # Next steps
            st.info("🚀 **Next Steps**: Visit the other pages to continue analysis:")
            st.write("1. 🔍 **Transit Detection**: View detailed period analysis")
            st.write("2. 🤖 **AI Classification**: Get ML predictions")
            st.write("3. 🔬 **Explainable AI**: Understand the model's reasoning")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.write("Please check that your file is a valid CSV with numeric time/flux data.")

# Use synthetic data option
elif 'synthetic_data' in st.session_state:
    st.subheader("🧪 Using Generated Synthetic Data")
    
    synth_df = st.session_state['synthetic_data']
    
    # Show preview
    st.dataframe(synth_df.head(), use_container_width=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(synth_df['time'], synth_df['flux'], '.', ms=1, alpha=0.7)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Flux')
    ax.set_title('Synthetic Light Curve')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    
    if st.button("🚀 Analyze Synthetic Data"):
        # Store for analysis
        time_clean = synth_df['time'].values
        flux_clean = synth_df['flux'].values
        
        st.session_state['light_curve_data'] = (time_clean, flux_clean)
        st.session_state['source_label'] = "Synthetic data"
        
        # Run BLS
        bls_results = simple_bls(time_clean, flux_clean, 10.0)
        st.session_state['bls_results'] = bls_results
        
        st.success("✅ Synthetic data ready for analysis!")

else:
    st.info("👆 Upload a CSV file or generate synthetic data to begin analysis.")

# Data format examples
st.subheader("📚 Data Format Examples")

with st.expander("💡 See example data formats"):
    st.write("**Example 1: Basic format**")
    st.code("""time,flux
0.0,1.001
0.02,0.999
0.04,1.002
0.06,0.998
...""")
    
    st.write("**Example 2: With error bars (error column ignored)**")
    st.code("""time,flux,flux_err
1234.567,1.0012,0.0003
1234.587,0.9998,0.0003
1234.607,1.0005,0.0003
...""")
    
    st.write("**Example 3: Different column names (first two numeric used)**")
    st.code("""mjd,normalized_flux,quality
58001.0,1.001,0
58001.02,0.999,0
58001.04,1.002,0
...""")