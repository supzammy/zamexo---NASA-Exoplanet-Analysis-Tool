# ZAMEXO

**Exploring Exoplanets with Real NASA Data**

---

`ZAMEXO` is a minimalist, open-source Streamlit app for exploring and analyzing exoplanet light curves from NASA missions (Kepler, K2, TESS) or your own data. It is designed for scientists, students, and enthusiasts who want to visualize and analyze real space telescope data with ease.

---

## Features

- **Fetch Real NASA Data:**  
  Enter a target name and select a mission to download and plot authentic NASA light curves using the [Lightkurve](https://docs.lightkurve.org/) library.
- **Automatic Fallback:**  
  The app tries multiple NASA data products for each target, increasing the chance of a successful download even if some files are corrupt or unavailable.
- **CSV Upload:**  
  Upload your own light curve data (CSV/TXT with two numeric columns) for instant plotting and analysis.
- **Interactive Visualization:**  
  Zoom in on time ranges, view basic statistics (mean, std, min, max, N), and see a Lomb-Scargle periodogram to search for periodic signals.
- **User-Friendly:**  
  Minimalist interface, clear error messages, and progress spinners for a smooth experience.
- **Open Science Ready:**  
  All code and dependencies are open source, supporting NASA’s open science and reproducibility goals.

---

## Quick Start

1. **Clone and install dependencies:**
    ```sh
    git clone <your-repo-url>
    cd zamexo
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Run the app:**
    ```sh
    streamlit run app/streamlit_app.py
    ```

---

## Usage

- **NASA Data:**  
  Enter a target name (e.g. `TRAPPIST-1`) and select a mission (`TESS`, `Kepler`, or `K2`).  
  Click "Fetch NASA light curve" to download and plot real data.
- **CSV Upload:**  
  Or, upload your own CSV/TXT file with two numeric columns (time, flux) via the sidebar.

---

## NASA Open Science Alignment

- **Open Data Access:**  
  Fetches and visualizes publicly available NASA mission data.
- **Transparency:**  
  Shows which NASA data products are accessed and any issues encountered.
- **User Empowerment:**  
  Allows users to analyze their own data if NASA servers are unavailable.
- **Reproducibility:**  
  All code and dependencies are open source and can be run locally.

---

## Troubleshooting

- **NASA data fetch fails:**  
  This is often due to corrupt or unavailable files on the NASA/MAST server. Try a different target or upload your own CSV.
- **App is slow:**  
  Large light curves are downsampled for speed, but NASA downloads can still take up to a minute.

---

## Example Screenshot

![screenshot](docs/screenshot.png)
 <!-- Screenshots will be shared later -->

---

## Credits

- [Lightkurve](https://docs.lightkurve.org/) for NASA data access
- [Streamlit](https://streamlit.io/) for the app framework

---

## License

MIT License 
