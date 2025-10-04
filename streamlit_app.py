import streamlit as st
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.title("NASA Light Curve Explorer")
st.markdown("### zamexo — Exploring Exoplanets with Real NASA Data")

target = st.sidebar.text_input("Target name (e.g. TRAPPIST-1)", value="TRAPPIST-1")
mission = st.sidebar.selectbox("Mission", ["TESS", "Kepler", "K2"], index=0)
fetch = st.sidebar.button("Fetch light curve")

lc = None

if fetch:
    with st.spinner("Downloading and processing light curve..."):
        try:
            search = lk.search_lightcurve(target, mission=mission)
            st.write(search)
            if len(search) > 0:
                for i, prod in enumerate(search[:3]):  # Try only first 3 products
                    try:
                        lc = prod.download()
                        st.success(f"Downloaded product #{i}: {prod}")
                        break
                    except Exception as e:
                        st.warning(f"Product #{i} failed: {e}")
                if lc is None:
                    st.error("All products failed to download. Try a different target or mission.")
            else:
                st.error("No products found for this target/mission.")
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.info("Try a different target or mission.")

if lc is not None:
    # Downsample for plotting
    if len(lc.time) > 10000:
        idx = np.linspace(0, len(lc.time) - 1, 10000).astype(int)
        t_plot = lc.time[idx]
        y_plot = lc.flux[idx]
    else:
        t_plot = lc.time
        y_plot = lc.flux
    fig, ax = plt.subplots()
    ax.plot(t_plot, y_plot, ".", markersize=2)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Flux")
    ax.set_title(f"{target} ({mission})")
    st.pyplot(fig)
else:
    st.info("Enter a target and click 'Fetch light curve' to begin.")

st.header("Or upload your own light curve CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv", "txt"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("CSV preview:", df.head())
        cols = df.select_dtypes(include='number').columns
        if len(cols) >= 2:
            t, y = df[cols[0]], df[cols[1]]
            fig, ax = plt.subplots()
            ax.plot(t, y, ".")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            st.pyplot(fig)
        else:
            st.error("CSV must have at least two numeric columns.")
    except Exception as e:
        st.error(f"CSV upload failed: {e}")

st.sidebar.info("NASA data download may take up to a minute.")


