import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

st.set_page_config(page_title="NASA Light Curve Explorer", layout="wide")

# ---------------- Shared Helpers ---------------- #
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

@st.cache_data(show_spinner=False)
def load_artifacts():
    model = None; feature_cols=[]; feature_stats={}; importances={}
    models_dir = Path('models')
    try:
        import joblib  # type: ignore
        mp = models_dir/'model.joblib'
        if mp.exists():
            model = joblib.load(mp)
    except Exception:
        model=None
    def _load_json(p):
        try:
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            return None
        return None
    feature_cols = _load_json(models_dir/'feature_cols.json') or []
    feature_stats = _load_json(models_dir/'feature_stats.json') or {}
    importances = _load_json(models_dir/'importances.json') or {}
    return model, feature_cols, feature_stats, importances

@st.cache_resource(show_spinner=False)
def _get_shap_explainer(_model):
    if not _HAS_SHAP or _model is None:
        return None
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

def run_inference(model, feature_cols, bls):
    if model is None or not feature_cols or bls is None:
        return None, {}, None
    # Map minimal BLS dict into feature row; unseen features zero-filled
    row = {}
    for c in feature_cols:
        v = bls.get(c, np.nan) if isinstance(bls, dict) else np.nan
        row[c] = v if isinstance(v, (int,float,np.floating)) else np.nan
    X = pd.DataFrame([row], columns=feature_cols).fillna(0)
    try:
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model,'classes_', [f'class_{i}' for i in range(len(proba))]))
        scores = dict(zip(classes, map(float, proba)))
        label = classes[int(np.argmax(proba))]
        return label, scores, X
    except Exception:
        return None, {}, X

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

def get_shap_plot(model, X):
    if not _HAS_SHAP or model is None or X is None or X.empty:
        return None
    expl = _get_shap_explainer(model)
    if expl is None:
        return None
    try:
        sv = expl.shap_values(X)
        proba = model.predict_proba(X)[0]
        cls = int(np.argmax(proba))
        fig, ax = plt.subplots(figsize=(6,2.2))
        shap.force_plot(expl.expected_value[cls], sv[cls], X.iloc[0], matplotlib=True, show=False, ax=ax)
        fig.tight_layout()
        return fig
    except Exception:
        return None

# ---------------- Mode Toggle ---------------- #
mode = st.sidebar.radio("UI Mode", ["Modern","Legacy"], index=0, help="Switch between new tabbed interface and original demo.")

if mode == "Legacy":
    st.title("NASA Light Curve Demo (Legacy Mode)")
    # (retain original logic below)
    
    target = st.sidebar.text_input("Target name", value="TRAPPIST-1")
    mission = st.sidebar.selectbox("Mission", ["TESS", "Kepler", "K2"], index=0)
    fetch = st.sidebar.button("Fetch NASA light curve")
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Or upload your own CSV", type=["csv", "txt"])
    lc = None; data_source=None
    if fetch:
        try:
            search = lk.search_lightcurve(target, mission=mission)
            st.write(search)
            if len(search)>0:
                for i,prod in enumerate(search):
                    try:
                        lc = prod.download(); st.success(f"Downloaded product #{i}: {prod}"); data_source=f"NASA: {target} ({mission})"; break
                    except Exception as e:
                        st.warning(f"Product #{i} failed: {e}")
                if lc is None:
                    st.error("All products failed to download. Try a different target or mission.")
            else:
                st.error("No products found for this target/mission.")
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
    if lc is None and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            cols = df.select_dtypes(include='number').columns
            if len(cols)>=2:
                t,y = df[cols[0]], df[cols[1]]; lc=(t,y); data_source=f"CSV: {uploaded_file.name}"; st.success("CSV loaded.")
            else:
                st.error("CSV must have at least two numeric columns.")
        except Exception as e:
            st.error(f"CSV upload failed: {e}")
    if lc is not None:
        st.subheader(f"Light Curve ({data_source})")
        if isinstance(lc, tuple):
            t,y = np.asarray(lc[0]), np.asarray(lc[1])
        else:
            t,y = lc.time.value, lc.flux.value
        stats = summarize_lightcurve(t,y)
        st.write({k:stats[k] for k in ['mean_flux','std_flux','n_points','baseline_days'] if k in stats})
        tmin,tmax=float(np.nanmin(t)), float(np.nanmax(t))
        tr=st.slider("Time range (days)", min_value=tmin, max_value=tmax, value=(tmin,tmax))
        m=(t>=tr[0])&(t<=tr[1])
        fig, ax=plt.subplots(figsize=(10,4))
        ax.plot(t[m], y[m], '.', ms=2)
        ax.set_xlabel('Time [days]'); ax.set_ylabel('Flux'); ax.set_title('Light Curve')
        st.pyplot(fig)
    else:
        st.info("Fetch a NASA light curve or upload a CSV to begin.")
    st.stop()

# ---------------- Modern Mode ---------------- #
st.title("🚀 NASA Transit Search & Explainable AI")
with st.sidebar:
    mission = st.selectbox("Mission", ["Auto","Kepler","TESS"], index=0)
    target = st.text_input("Target (e.g., Kepler-10)", "")
    max_period = st.slider("Max period [days]", 1.0, 50.0, 10.0, step=0.5)
    uploaded = st.file_uploader("Upload CSV (time,flux)", type=['csv'])
    run_btn = st.button("Run Analysis")
    enable_shap = st.checkbox("Enable SHAP", value=False)
    if enable_shap and not _HAS_SHAP:
        st.warning("SHAP not installed in environment.")

model, feature_cols, feature_stats, importances = load_artifacts()

time_arr = flux_arr = None; source_label="(none)"
if uploaded is not None:
    try:
        df_u = pd.read_csv(uploaded)
        if {'time','flux'}.issubset(df_u.columns):
            time_arr = df_u['time'].to_numpy(float); flux_arr=df_u['flux'].to_numpy(float); source_label=f"Upload: {uploaded.name}"
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
if time_arr is None and run_btn and target.strip():
    target_clean = target.strip(); missions = [mission] if mission!="Auto" else ["Kepler","K2","TESS"]
    for m in missions:
        try:
            sr = lk.search_lightcurve(target_clean, mission=m)
            if len(sr)==0: continue
            parts=[]
            for prod in sr[:3]:
                try:
                    lc_part = prod.download();
                    if lc_part is not None: parts.append(lc_part)
                except Exception: pass
            if not parts: continue
            lc_full = parts[0]
            if len(parts)>1:
                try: lc_full = lc_full.append(parts[1:])
                except Exception: pass
            tvals = lc_full.time.value if hasattr(lc_full.time,'value') else np.asarray(lc_full.time)
            fvals = lc_full.flux.value if hasattr(lc_full.flux,'value') else np.asarray(lc_full.flux)
            mask = np.isfinite(tvals) & np.isfinite(fvals)
            time_arr, flux_arr = tvals[mask], fvals[mask]
            source_label = f"NASA: {target_clean} ({m})"; break
        except Exception: continue
if time_arr is None:
    rng = np.random.default_rng(42)
    time_arr = np.linspace(0,12,3000); flux_arr = 1 + 0.001*np.sin(2*np.pi*time_arr/2.1) + 5e-4*rng.standard_normal(time_arr.size)
    source_label = "Synthetic demo"

bls = simple_bls(time_arr, flux_arr, max_period=max_period)
depth_ppm = bls['depth']*1e6 if np.isfinite(bls.get('depth', np.nan)) else np.nan

tabs = st.tabs(["Overview","Detection","AI Classification","Explainability","Upload"])
with tabs[0]:
    st.subheader("Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Period [d]", f"{bls['period']:.4g}" if np.isfinite(bls['period']) else "–")
    c2.metric("Duration [d]", f"{bls['duration']:.4g}" if np.isfinite(bls['duration']) else "–")
    c3.metric("Depth", f"{bls['depth']:.3g}" if np.isfinite(bls['depth']) else "–", help=f"≈{depth_ppm:.0f} ppm" if np.isfinite(depth_ppm) else None)
    c4.metric("SDE", f"{bls['sde']:.3g}" if np.isfinite(bls['sde']) else "–")
    st.caption(f"Source: {source_label}")
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(time_arr, flux_arr, '.', ms=2, alpha=0.7)
    ax.set_xlabel('Time [d]'); ax.set_ylabel('Flux'); ax.grid(alpha=0.3)
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Detection")
    if np.isfinite(bls['period']):
        phase = ((time_arr - time_arr.min()) % bls['period'])/bls['period']
        order = np.argsort(phase)
        figp, axp = plt.subplots(figsize=(9,3))
        axp.plot(phase[order], flux_arr[order], '.', ms=2, alpha=0.6)
        axp.set_xlabel('Phase'); axp.set_ylabel('Flux'); axp.grid(alpha=0.3)
        st.pyplot(figp)
    else:
        st.info("No reliable period candidate.")

with tabs[2]:
    st.subheader("AI Classification")
    label, probs, X_pred = run_inference(model, feature_cols, bls)
    if model is None:
        st.warning("Model artifacts not found (train via scripts/train_baseline.py).")
    elif label is None:
        st.info("Not enough features for prediction.")
    else:
        top = max(probs, key=probs.get) if probs else label
        st.markdown(f"**Prediction:** `{label}` | Confidence: {probs.get(top,0):.2%}")
        if probs:
            st.bar_chart(pd.Series(probs))

with tabs[3]:
    st.subheader("Explainability")
    if enable_shap:
        label, probs, X_pred = run_inference(model, feature_cols, bls)
        if X_pred is None or not _HAS_SHAP:
            st.info("SHAP not available or no prediction context.")
        else:
            with st.spinner("Computing SHAP..."):
                shap_fig = get_shap_plot(model, X_pred)
            if shap_fig:
                st.pyplot(shap_fig)
            else:
                st.warning("Unable to generate SHAP plot.")
    else:
        st.info("Enable SHAP in the sidebar to view per‑prediction contributions.")

with tabs[4]:
    st.subheader("Upload Workflow")
    st.write("Upload a CSV with columns 'time','flux' in sidebar. It feeds the entire pipeline.")
    stats = summarize_lightcurve(time_arr, flux_arr)
    if stats:
        colA,colB,colC,colD = st.columns(4)
        colA.metric("Points", stats['n_points'])
        colB.metric("Baseline (d)", f"{stats['baseline_days']:.2f}")
        colC.metric("Cadence (s)", f"{stats['cadence_days']*86400:.0f}" if np.isfinite(stats['cadence_days']) else "–")
        colD.metric("Frac RMS", f"{stats['frac_rms']:.2%}" if np.isfinite(stats['frac_rms']) else "–")
    st.caption("Legacy mode still available from the sidebar toggle.")