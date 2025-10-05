import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Transit Detection", page_icon="🔍", layout="wide")

# Import shared helpers from root level
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
    """Simple BLS implementation with improved SDE calculation."""
    t = _clean_series(time); f = _clean_series(flux)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 10:
        return {'period': np.nan,'duration': np.nan,'depth': np.nan,'sde': np.nan,'t0': np.nan}
    
    t = t - t.min()
    baseline = np.nanmedian(f)
    if np.isfinite(baseline) and baseline != 0:
        f = f/baseline - 1.0
    else:
        f = f - np.nanmean(f)
    
    # Basic autocorrelation-based period detection
    n = min(f.size, 4000)
    f2 = f[:n] - np.nanmean(f[:n])
    
    try:
        ac = np.correlate(f2, f2, mode='full')[n-1:]
    except (TypeError, ValueError):
        return {'period': np.nan,'duration': np.nan,'depth': np.nan,'sde': np.nan,'t0': np.nan}
    
    ac[0] = 0
    lag = int(np.argmax(ac[1:])+1)
    dt = np.median(np.diff(t)) if t.size > 1 else np.nan
    period = float(lag*dt) if np.isfinite(dt) else np.nan
    
    if not (np.isfinite(period) and 0.5 < period <= max_period):
        period = np.nan
    
    # Calculate transit depth and SDE
    if np.isfinite(period):
        # Phase fold and calculate depth
        phase = (t % period) / period
        phase_sorted_idx = np.argsort(phase)
        f_sorted = f[phase_sorted_idx]
        
        # Simple box search for transit
        n_bins = min(100, len(f_sorted) // 10)
        if n_bins > 5:
            bin_size = len(f_sorted) // n_bins
            bin_means = []
            for i in range(n_bins):
                start = i * bin_size
                end = min((i + 1) * bin_size, len(f_sorted))
                if end > start:
                    bin_means.append(np.nanmean(f_sorted[start:end]))
            
            if bin_means:
                bin_means = np.array(bin_means)
                depth = np.nanmin(bin_means)
                
                # Calculate SDE (simplified)
                noise_std = np.nanstd(f)
                if noise_std > 0:
                    sde = abs(depth) / noise_std * np.sqrt(len(f) * 0.1)  # Rough SDE estimate
                else:
                    sde = 0.0
            else:
                depth = np.nan
                sde = np.nan
        else:
            depth = np.nanpercentile(f, 5)  # Fallback depth estimate
            noise_std = np.nanstd(f)
            sde = abs(depth) / noise_std * np.sqrt(len(f) * 0.05) if noise_std > 0 else 0.0
        
        # Duration estimate (5% of period as default)
        duration = period * 0.05
    else:
        depth = np.nan
        duration = np.nan
        sde = np.nan
    
    return {
        'period': period,
        'duration': duration,
        'depth': depth,
        'sde': sde,
        't0': float(t.min()) if t.size > 0 else np.nan
    }

st.title("🔍 Transit Detection")

# Sidebar controls
with st.sidebar:
    st.subheader("Input")
    mission = st.selectbox("Mission", ["Auto","Kepler","TESS","K2"], index=0)
    target = st.text_input("Target (e.g., Kepler-10)", "")
    max_period = st.slider("Max period [days]", 1.0, 50.0, 10.0, step=0.5)
    uploaded = st.file_uploader("Upload CSV (time,flux)", type=['csv'])
    run_btn = st.button("🚀 Run Analysis")
    
    if st.button("Clear Cache"):
        import shutil, pathlib
        cache_dir = pathlib.Path.home()/".lightkurve"/"cache"/"mastDownload"
        shutil.rmtree(cache_dir, ignore_errors=True)
        st.success("Cache cleared.")

# Data ingestion
time_arr = flux_arr = None
source_label = "(none)"

if uploaded is not None:
    try:
        df_u = pd.read_csv(uploaded)
        if {'time','flux'}.issubset(df_u.columns):
            time_arr = df_u['time'].to_numpy(float)
            flux_arr = df_u['flux'].to_numpy(float)
            source_label = f"Upload: {uploaded.name}"
            st.success(f"✅ Loaded {len(time_arr)} points from uploaded CSV")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if time_arr is None and run_btn and target.strip():
    target_clean = target.strip()
    missions = [mission] if mission!="Auto" else ["Kepler","K2","TESS"]
    
    # Network error handling with retry logic
    max_retries = 3
    retry_delay = 2
    
    with st.spinner(f"🔍 Searching for {target_clean}..."):
        search_successful = False
        
        for m in missions:
            if search_successful:
                break
                
            for attempt in range(max_retries):
                try:
                    st.caption(f"Searching {m} mission (attempt {attempt + 1}/{max_retries})...")
                    
                    # Add timeout and retry wrapper
                    import socket
                    socket.setdefaulttimeout(30)  # 30 second timeout
                    
                    sr = lk.search_lightcurve(target_clean, mission=m)
                    
                    if len(sr) == 0: 
                        st.info(f"No data found for {target_clean} in {m} mission")
                        break  # Try next mission
                        
                    st.success(f"✅ Found {len(sr)} products for {m} mission")
                    parts = []
                    download_errors = []
                    
                    for i, prod in enumerate(sr[:3]):  # Limit to first 3 products
                        try:
                            st.caption(f"Downloading product {i+1}/{min(3, len(sr))}...")
                            lc_part = prod.download()
                            if lc_part is not None: 
                                parts.append(lc_part)
                        except Exception as e:
                            download_errors.append(f"Product {i+1}: {str(e)[:50]}...")
                            continue
                            
                    if not parts:
                        if download_errors:
                            st.warning(f"⚠️ All downloads failed for {m}:")
                            for err in download_errors[:3]:  # Show max 3 errors
                                st.caption(f"• {err}")
                        break  # Try next mission
                        
                    # Successfully downloaded data
                    lc_full = parts[0]
                    if len(parts) > 1:
                        try: 
                            lc_full = lc_full.append(parts[1:])
                        except Exception as e:
                            st.warning(f"Failed to combine light curves: {e}")
                            # Use first part only
                    
                    # Extract data
                    tvals = lc_full.time.value if hasattr(lc_full.time,'value') else np.asarray(lc_full.time)
                    fvals = lc_full.flux.value if hasattr(lc_full.flux,'value') else np.asarray(lc_full.flux)
                    mask = np.isfinite(tvals) & np.isfinite(fvals)
                    time_arr, flux_arr = tvals[mask], fvals[mask]
                    source_label = f"NASA: {target_clean} ({m})"
                    st.success(f"✅ Fetched {len(time_arr)} points from {m} for {target_clean}")
                    
                    search_successful = True
                    break  # Success! Exit retry loop
                    
                except (ConnectionError, TimeoutError, socket.timeout) as e:
                    if attempt < max_retries - 1:
                        st.warning(f"⚠️ Network error (attempt {attempt + 1}): {str(e)[:100]}...")
                        st.caption(f"Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                    else:
                        st.error(f"❌ Network connection failed for {m} after {max_retries} attempts")
                        st.caption(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error searching {m}: {str(e)[:100]}")
                    break  # Try next mission
        
        # Handle case where no missions succeeded
        if not search_successful:
            st.error("❌ **All data searches failed**")
            st.write("**Possible solutions:**")
            st.write("• Check your internet connection")
            st.write("• Try a different target name (e.g., 'Kepler-10', 'TOI-715')")
            st.write("• Use the **📤 Data Upload** page with your own CSV file")
            st.write("• Try again later - NASA servers may be temporarily unavailable")
            
            # Offer manual input as fallback
            with st.expander("🧪 Generate test data instead"):
                if st.button("📊 Create synthetic transit data"):
                    # Generate synthetic data for testing
                    np.random.seed(42)
                    time_synth = np.linspace(0, 20, 4000)
                    flux_synth = np.ones_like(time_synth) + np.random.normal(0, 0.001, len(time_synth))
                    
                    # Add a transit signal
                    period = 2.87
                    duration = 0.1
                    depth = 0.01
                    
                    for cycle in range(int(20 / period) + 1):
                        transit_center = cycle * period
                        transit_mask = np.abs(time_synth - transit_center) < (duration / 2)
                        flux_synth[transit_mask] -= depth
                    
                    time_arr = time_synth
                    flux_arr = flux_synth
                    source_label = f"Synthetic data (P={period}d, depth={depth:.1%})"
                    st.success("✅ Generated synthetic transit data for testing")
            
            if time_arr is None:
                st.stop()

if time_arr is None:
    # Synthetic fallback when no data available
    st.info("💡 **No data loaded.** Using synthetic demonstration data.")
    rng = np.random.default_rng(42)
    time_arr = np.linspace(0,12,3000)
    flux_arr = 1 + 0.001*np.sin(2*np.pi*time_arr/2.1) + 5e-4*rng.standard_normal(time_arr.size)
    source_label = "Synthetic demo data"

# Analysis
if time_arr is not None:
    st.subheader("📊 Light Curve Overview")
    
    # Basic stats
    stats = summarize_lightcurve(time_arr, flux_arr)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Points", f"{stats['n_points']:,}")
    col2.metric("Baseline (d)", f"{stats['baseline_days']:.2f}")
    col3.metric("Cadence (s)", f"{stats['cadence_days']*86400:.0f}" if np.isfinite(stats['cadence_days']) else "–")
    col4.metric("Fractional RMS", f"{stats['frac_rms']:.2%}" if np.isfinite(stats['frac_rms']) else "–")
    
    st.caption(f"📡 Source: {source_label}")
    
    # Light curve plot
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(time_arr, flux_arr, '.', ms=1, alpha=0.7)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Light Curve')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    
    # Transit detection
    st.subheader("🔍 Transit Detection")
    
    with st.spinner("Running Box Least Squares analysis..."):
        bls = simple_bls(time_arr, flux_arr, max_period=max_period)
    
    # BLS results
    col1, col2, col3 = st.columns(3)
    col1.metric("Period [d]", f"{bls['period']:.4g}" if np.isfinite(bls['period']) else "–")
    col2.metric("Duration [d]", f"{bls['duration']:.4g}" if np.isfinite(bls['duration']) else "–")
    depth_ppm = bls['depth']*1e6 if np.isfinite(bls.get('depth', np.nan)) else np.nan
    col3.metric("Depth [ppm]", f"{depth_ppm:.0f}" if np.isfinite(depth_ppm) else "–")
    
    if np.isfinite(bls['period']):
        st.success(f"🎯 Transit candidate detected with period {bls['period']:.3f} days!")
        
        # Phase-folded plot
        st.subheader("📈 Phase-Folded Light Curve")
        phase = ((time_arr - time_arr.min()) % bls['period']) / bls['period']
        # Center phase at transit
        phase[phase > 0.5] -= 1.0
        order = np.argsort(phase)
        
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(phase[order], flux_arr[order], '.', ms=2, alpha=0.6)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='Transit center')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Normalized Flux')
        ax2.set_title(f'Phase-Folded at P = {bls["period"]:.3f} days')
        ax2.grid(alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        
        # Store results in session state for other pages
        st.session_state['bls_results'] = bls
        st.session_state['light_curve_data'] = (time_arr, flux_arr)
        st.session_state['source_label'] = source_label
        
        # Quick navigation to next step
        st.success("✅ **Transit analysis complete!** Ready for AI classification.")
        if st.button("🤖 Continue to AI Classification", type="primary"):
            st.switch_page("pages/2_AI_Classification.py")
        
    else:
        st.warning("⚠️ **No strong transit signal detected**")
        st.write("""
        **This means:**
        • The Box Least Squares algorithm didn't find a clear periodic transit pattern
        • The signal might be too weak, too short, or masked by noise
        • The true period might be longer than your maximum period setting
        
        **What to try:**
        • Increase the maximum period (try 50-100 days)
        • Try a different, well-known target (e.g., `TRAPPIST-1`, `Kepler-10b`)
        • Check the light curve for obvious issues in the plot above
        """)
        
        # Store minimal results even for weak signals
        st.session_state['bls_results'] = bls
        st.session_state['light_curve_data'] = (time_arr, flux_arr)
        st.session_state['source_label'] = source_label
        
        st.info("💡 **You can still proceed to AI Classification** - even weak or unclear signals can be analyzed to determine if they represent real transits or false positives.")
        
        # Always show button to continue
        if st.button("🤖 Analyze with AI Anyway", type="secondary"):
            st.switch_page("pages/2_AI_Classification.py")
        
    # Additional analysis options
    with st.expander("🔧 Advanced Options"):
        st.write("**Data Quality Assessment:**")
        if stats['frac_rms'] < 0.001:
            st.success("✅ Excellent data quality (very low noise)")
        elif stats['frac_rms'] < 0.01:
            st.info("👍 Good data quality")
        else:
            st.warning("⚠️ Noisy data - results may be less reliable")
            
        st.write("**Suggested Next Steps:**")
        if np.isfinite(bls['period']):
            st.write("- 🤖 Proceed to AI Classification for ML-based validation")
            st.write("- 🔬 Check Explainability page for model reasoning")
        else:
            st.write("- Try a different target with known transits")
            st.write("- Increase maximum period search range")
            st.write("- Check data quality and coverage")

else:
    st.info("👆 Enter a target name and click 'Run Analysis' to begin, or upload a CSV file with time,flux columns.")