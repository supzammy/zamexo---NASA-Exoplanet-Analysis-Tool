import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(page_title="NASA Light Curve Demo", layout="wide")
st.title("NASA Light Curve Demo")

# --- Sidebar ---
target = st.sidebar.text_input("Target name", value="TRAPPIST-1")
mission = st.sidebar.selectbox("Mission", ["TESS", "Kepler", "K2"], index=0)
fetch = st.sidebar.button("Fetch NASA light curve")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV", type=["csv", "txt"])

lc = None
data_source = None

# --- Try NASA fetch ---
if fetch:
    try:
        search = lk.search_lightcurve(target, mission=mission)
        st.write(search)
        if len(search) > 0:
            for i, prod in enumerate(search):
                try:
                    lc = prod.download()
                    st.success(f"Downloaded product #{i}: {prod}")
                    data_source = f"NASA: {target} ({mission})"
                    break
                except Exception as e:
                    st.warning(f"Product #{i} failed: {e}")
            if lc is None:
                st.error("All products failed to download. Try a different target or mission.")
        else:
            st.error("No products found for this target/mission.")
    except Exception as e:
        st.error(f"Data fetch failed: {e}")

# --- Try CSV upload ---
if lc is None and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        cols = df.select_dtypes(include='number').columns
        if len(cols) >= 2:
            t, y = df[cols[0]], df[cols[1]]
            lc = (t, y)
            data_source = f"CSV: {uploaded_file.name}"
            st.success("CSV loaded.")
        else:
            st.error("CSV must have at least two numeric columns.")
    except Exception as e:
        st.error(f"CSV upload failed: {e}")

# --- Plot and Analysis ---
if lc is not None:
    st.subheader(f"Light Curve ({data_source})")
    if isinstance(lc, tuple):  # CSV
        t, y = np.asarray(lc[0]), np.asarray(lc[1])
    else:  # NASA
        t, y = lc.time.value, lc.flux.value

    # Basic stats
    st.write({
        "Mean flux": float(np.nanmean(y)),
        "Std flux": float(np.nanstd(y)),
        "Min flux": float(np.nanmin(y)),
        "Max flux": float(np.nanmax(y)),
        "N points": len(y)
    })

    # Zoom slider
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    time_range = st.slider("Time range (days)", min_value=tmin, max_value=tmax, value=(tmin, tmax))
    mask = (t >= time_range[0]) & (t <= time_range[1])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[mask], y[mask], ".", markersize=2)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Flux")
    ax.set_title("Light Curve")
    st.pyplot(fig)

    # Periodogram
    st.subheader("Periodogram (Lomb-Scargle)")
    try:
        if isinstance(lc, tuple):
            from astropy.timeseries import LombScargle
            freq, power = LombScargle(t[mask], y[mask]).autopower()
            period = 1 / freq
            best_period = period[np.argmax(power)]
        else:
            periodogram = lc[mask].to_periodogram(method="lombscargle", minimum_period=0.5, maximum_period=20)
            period, power = periodogram.period.value, periodogram.power.value
            best_period = periodogram.period_at_max_power.value

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(period, power, color="C1")
        ax2.set_xlabel("Period [days]")
        ax2.set_ylabel("Power")
        ax2.set_title("Lomb-Scargle Periodogram")
        st.pyplot(fig2)
        st.info(f"Best period: {best_period:.4f} days")
    except Exception as e:
        st.warning(f"Periodogram failed: {e}")

else:
    st.info("Fetch a NASA light curve or upload a CSV to begin.")