<img width="1176" height="702" alt="Screenshot 2025-11-08 at 9 25 59 PM" src="https://github.com/user-attachments/assets/d78cde3a-81cf-4bcb-9f1f-8f17b9871731" /># ZAMEXO

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

**Landing Page:**
  
<img width="1439" height="751" alt="Screenshot 2025-11-07 at 5 06 34 AM" src="https://github.com/user-attachments/assets/6f67fd03-4cc2-433e-8508-58750c0b7d91" />
<img width="1070" height="698" alt="Screenshot 2025-11-07 at 5 06 48 AM" src="https://github.com/user-attachments/assets/a12ff367-98f6-4789-ad27-e90b4aab363e" />
<img width="1140" height="701" alt="<img width="1175" height="743" alt="Screenshot 2025-11-08 at 9 15 59 PM" src="https://github.com/user-attachments/assets/98ee819a-49fd-4a26-9b36-bb49fc373018" />
Screenshot 2025-11-07 at 5 07 12 AM" src="https://github.com/user-attachments/assets/4b6e8ff0-be43-4a28-bded-ca27a4cd3d5b" />
<img width="1176" height="702" alt="Screenshot 2025-11-08 at 9 25 59 PM" src="https://github.com/user-attachments/assets/7a8d9972-5f89-4062-96d3-d9ab676016ed" />


**Dashboard:**
 - <img width="1439" height="746" alt="Screenshot 2025-11-08 at 9 26 55 PM" src="https://github.com/user-attachments/assets/75380042-f2dc-43d0-87e9-c18233ee0bfa" />


**Wokring Dashboard:**
   
<img width="1035" height="600" alt="Screenshot 2025-11-08 at 9 20 20 PM" src="https://github.com/user-attachments/assets/e4fd2d68-560e-4dab-a8a6-1301c0d4390b" />
<img width="1040" height="724" alt="Screenshot 2025-11-08 at 9 20 02 PM" src="https://github.com/user-attachments/assets/2a0a6302-4aed-460c-908c-a22a0caabc1b" />
<img width="1006" height="671" alt="Screenshot 2025-11-08 at 9 19 53 PM" src="https://github.com/user-attachments/assets/fbb66a10-764b-45f2-977e-c2bf71f40ef7" />
<img width="1076" height="698" alt="Screenshot 2025-11-08 at 9 19 46 PM" src="https://github.com/user-attachments/assets/1a22601f-c6d2-4f3e-a6e7-25e5fe07060b" />
<img width="1439" height="746" alt="Screenshot 2025-11-08 at 9 26 55 PM" src="https://github.com/user-attachments/assets/5f6b590a-ad6c-4bf1-95c6-7cc1f85f6ddc" />


---

## Credits

- [Lightkurve](https://docs.lightkurve.org/) for NASA data access
- [Streamlit](https://streamlit.io/) for the app framework

---

## License

MIT License 
