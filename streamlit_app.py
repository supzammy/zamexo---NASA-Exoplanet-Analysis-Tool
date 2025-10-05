import streamlit as st

# Redirect notice for Streamlit Cloud compatibility
st.set_page_config(
    page_title="ZAMEXO - Redirect",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
# 🚀 ZAMEXO - NASA Exoplanet Analysis Tool

## ⚠️ **Important Notice**
This app has been upgraded to a **multipage application**!

### 📌 **To access the full ZAMEXO experience:**
1. **If you're on Streamlit Cloud**: The main app should be at `Home.py`
2. **If you're running locally**: Use `streamlit run Home.py`

### 🔄 **Automatic Redirect**
The new multipage app includes:
- 🏠 **Home** - Professional interface and overview
- 🔍 **Transit Detection** - NASA data fetching and BLS analysis  
- 🤖 **AI Classification** - Enhanced ML predictions with proper feature mapping
- 🔬 **Explainable AI** - SHAP explanations and feature importance
- 📤 **Data Upload** - CSV upload and analysis workflow
- ⚙️ **Settings** - Configuration and system information

---
*This redirect page exists for Streamlit Cloud compatibility. Please update your deployment to use `Home.py` as the main file.*
""")

# Add a basic redirect button
if st.button("🚀 **Go to ZAMEXO Multipage App**", type="primary"):
    st.markdown("Please navigate manually to the multipage app or update your Streamlit Cloud deployment to use `Home.py`")

# Display current structure
with st.expander("📁 Current App Structure"):
    st.code("""
NASA Project/
├── Home.py                    ← NEW MAIN ENTRY POINT
├── pages/
│   ├── 1_🔍_Transit_Detection.py
│   ├── 2_🤖_AI_Classification.py
│   ├── 3_🔬_Explainable_AI.py
│   ├── 4_📤_Data_Upload.py
│   └── 5_⚙️_Settings.py
└── streamlit_app.py          ← OLD ENTRY POINT (THIS FILE)
    """)