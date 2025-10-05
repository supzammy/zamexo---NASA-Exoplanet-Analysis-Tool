import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X has feature names, but RandomForestClassifier was fitted without feature names")

st.set_page_config(page_title="AI Classification", page_icon="🤖", layout="wide")

# Import path setup
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load ML model and metadata."""
    model = None
    feature_cols = []
    feature_stats = {}
    importances = {}
    
    models_dir = Path('models')
    try:
        import joblib  # type: ignore
        mp = models_dir/'model.joblib'
        if mp.exists():
            model = joblib.load(mp)
    except Exception:
        model = None
        
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

def run_inference(model, feature_cols, bls):
    """Run ML inference on BLS features with proper feature mapping and defaults."""
    if model is None or not feature_cols or bls is None:
        return None, {}, None
        
    # Load feature statistics for better defaults
    import json
    from pathlib import Path
    
    models_dir = Path('models')
    feature_stats = {}
    try:
        stats_file = models_dir / 'feature_stats.json'
        if stats_file.exists():
            feature_stats = json.loads(stats_file.read_text())
    except Exception:
        pass
    
    # Create feature mapping with intelligent defaults
    row = {}
    for c in feature_cols:
        if c == 'koi_period':
            # Map BLS period to KOI period
            row[c] = bls.get('period', np.nan)
        elif c == 'koi_duration':
            # Map BLS duration to KOI duration (convert from days to hours if needed)
            duration = bls.get('duration', np.nan)
            row[c] = duration * 24.0 if np.isfinite(duration) else np.nan  # Convert to hours
        elif c == 'koi_depth':
            # Map BLS depth to KOI depth
            row[c] = bls.get('depth', np.nan)
        elif c == 'koi_impact':
            # Impact parameter - use reasonable default for transit
            row[c] = 0.5  # Moderate impact parameter
        elif c in ['koi_prad', 'koi_steff', 'koi_slogg', 'koi_srad']:
            # Use median values from training data for stellar parameters
            median_vals = feature_stats.get('median', {})
            row[c] = median_vals.get(c, np.nan)
        else:
            # Default fallback
            v = bls.get(c, np.nan) if isinstance(bls, dict) else np.nan
            row[c] = v if isinstance(v, (int,float,np.floating)) else np.nan
    
    # Create DataFrame with proper feature alignment
    X = pd.DataFrame([row], columns=feature_cols)
    
    # Fill remaining NaNs with median values from training data
    median_vals = feature_stats.get('median', {})
    for col in X.columns:
        if X[col].isna().any():
            default_val = median_vals.get(col, 0.0)
            X[col] = X[col].fillna(default_val)
    
    try:
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model,'classes_', [f'class_{i}' for i in range(len(proba))]))
        scores = dict(zip(classes, map(float, proba)))
        
        # Enhanced false positive detection logic
        max_prob = max(proba)
        pred_idx = int(np.argmax(proba))
        label = classes[pred_idx]
        
        # Apply additional validation for false positive detection
        period = bls.get('period', np.nan)
        depth = bls.get('depth', np.nan)
        duration = bls.get('duration', np.nan)
        sde = bls.get('sde', np.nan)
        
        # Red flags for false positives - more conservative thresholds
        false_positive_score = 0
        warning_flags = []
        
        # Period validation
        if np.isfinite(period):
            if period < 0.5:
                false_positive_score += 0.4
                warning_flags.append(f"Very short period ({period:.2f}d)")
            elif period > 100:
                false_positive_score += 0.3
                warning_flags.append(f"Very long period ({period:.1f}d)")
        
        # Depth validation - more realistic thresholds
        if np.isfinite(depth):
            depth_ppm = depth * 1e6
            if depth_ppm < 10:  # < 10 ppm is very shallow
                false_positive_score += 0.3
                warning_flags.append(f"Very shallow transit ({depth_ppm:.1f} ppm)")
            elif depth_ppm > 100000:  # > 10% is unrealistically deep
                false_positive_score += 0.4
                warning_flags.append(f"Unrealistically deep transit ({depth_ppm:.0f} ppm)")
        
        # Duration validation
        if np.isfinite(period) and np.isfinite(duration):
            duration_ratio = duration / period
            if duration_ratio > 0.3:  # Transit longer than 30% of period
                false_positive_score += 0.5
                warning_flags.append(f"Transit too long ({duration_ratio:.1%} of period)")
            elif duration_ratio < 0.001:  # Extremely short transit
                false_positive_score += 0.3
                warning_flags.append(f"Extremely short transit ({duration_ratio:.3%} of period)")
        
        # SDE validation
        if np.isfinite(sde):
            if sde < 5.0:
                false_positive_score += 0.2
                warning_flags.append(f"Low signal strength (SDE={sde:.1f})")
        
        # Store warning flags for display
        st.session_state['validation_warnings'] = warning_flags
        
        # Apply false positive adjustment if score is high
        if false_positive_score > 0.4 and 'FALSE POSITIVE' in classes:
            fp_idx = classes.index('FALSE POSITIVE')
            # Boost false positive probability more aggressively
            boost = min(0.6, false_positive_score * 0.8)
            proba[fp_idx] = min(0.98, proba[fp_idx] + boost)
            
            # Reduce other probabilities proportionally
            proba = proba / proba.sum()
            
            # Update scores and label
            scores = dict(zip(classes, map(float, proba)))
            pred_idx = int(np.argmax(proba))
            label = classes[pred_idx]
        
        return label, scores, X
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, {}, X

st.title("🤖 AI Classification")

# Load model artifacts
model, feature_cols, feature_stats, importances = load_artifacts()

if model is None:
    st.error("❌ **Model not found!**")
    st.write("Please train the model first:")
    st.code("python scripts/train_baseline.py", language="bash")
    st.info("The model will be saved to the `models/` directory.")
    st.stop()

# Check for BLS results from Transit Detection page
if 'bls_results' not in st.session_state:
    st.warning("⚠️ **No transit detection results found.**")
    st.write("Please visit the **🔍 Transit Detection** page first to analyze a light curve, or upload data in the **📤 Data Upload** page.")
    
    # Quick navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Go to Transit Detection", type="primary", use_container_width=True):
            st.switch_page("pages/1_🔍_Transit_Detection.py")
    with col2:
        if st.button("📤 Go to Data Upload", use_container_width=True):
            st.switch_page("pages/4_📤_Data_Upload.py")
    
    st.info("💡 **Tip**: The AI classifier needs transit features (period, depth, duration) to make predictions.")
    
    # Option to manually input features for testing
    with st.expander("🧪 Manual Feature Input (for testing)"):
        st.write("Enter transit features manually:")
        period = st.number_input("Period [days]", value=2.5, min_value=0.1, max_value=100.0)
        duration = st.number_input("Duration [days]", value=0.1, min_value=0.001, max_value=1.0)
        depth = st.number_input("Depth [fraction]", value=0.01, min_value=0.0001, max_value=0.1, format="%.4f")
        
        if st.button("🔍 Classify Manual Input"):
            manual_bls = {
                'period': period,
                'duration': duration, 
                'depth': depth,
                'sde': 5.0,  # dummy
                't0': 0.0   # dummy
            }
            st.session_state['bls_results'] = manual_bls
            st.session_state['source_label'] = "Manual input"
            st.rerun()
    
    st.stop()

# Get BLS results
bls = st.session_state['bls_results']
source_label = st.session_state.get('source_label', 'Unknown')

st.subheader("📊 Input Features")

# Display the features going into the model
col1, col2, col3, col4 = st.columns(4)
col1.metric("Period [d]", f"{bls.get('period', np.nan):.4g}" if np.isfinite(bls.get('period', np.nan)) else "–")
col2.metric("Duration [d]", f"{bls.get('duration', np.nan):.4g}" if np.isfinite(bls.get('duration', np.nan)) else "–")
depth_ppm = bls.get('depth', np.nan) * 1e6 if np.isfinite(bls.get('depth', np.nan)) else np.nan
col3.metric("Depth [ppm]", f"{depth_ppm:.0f}" if np.isfinite(depth_ppm) else "–")
col4.metric("SDE", f"{bls.get('sde', np.nan):.2f}" if np.isfinite(bls.get('sde', np.nan)) else "–")

st.caption(f"📡 Source: {source_label}")

# Show feature mapping information
with st.expander("🔧 Feature Mapping Details"):
    st.write("**How BLS results map to model features:**")
    
    mapping_info = {
        "Transit Features (from BLS)": [
            f"period → koi_period: {bls.get('period', 'N/A'):.4g}" if np.isfinite(bls.get('period', np.nan)) else "period → koi_period: N/A",
            f"duration → koi_duration: {bls.get('duration', np.nan)*24:.2f} hours" if np.isfinite(bls.get('duration', np.nan)) else "duration → koi_duration: N/A",
            f"depth → koi_depth: {bls.get('depth', np.nan):.6f}" if np.isfinite(bls.get('depth', np.nan)) else "depth → koi_depth: N/A"
        ],
        "Stellar Features (median defaults)": [
            "koi_steff: ~5761 K (stellar temperature)",
            "koi_slogg: ~4.44 (surface gravity)",
            "koi_srad: ~1.0 R☉ (stellar radius)",
            "koi_prad: ~2.5 R⊕ (planet radius estimate)"
        ]
    }
    
    for category, features in mapping_info.items():
        st.write(f"**{category}:**")
        for feature in features:
            st.write(f"• {feature}")
    
    st.info("💡 **Note**: Missing stellar parameters use median values from the training dataset. For better accuracy, provide stellar properties or use confirmed exoplanet hosts.")

# Run ML prediction
st.subheader("🎯 ML Prediction")

with st.spinner("🤖 Running AI classification..."):
    label, probs, X_pred = run_inference(model, feature_cols, bls)

if label is None:
    st.error("❌ Classification failed - insufficient features")
    st.stop()

# Store prediction results for Explainable AI page
st.session_state['ml_prediction'] = {
    'label': label,
    'probabilities': probs,
    'features': X_pred.values.tolist() if X_pred is not None else None,
    'feature_names': feature_cols
}

# Display prediction results
max_class = max(probs, key=probs.get) if probs else label
confidence = probs.get(max_class, 0.0) if probs else 0.0

# Enhanced color-coding and messaging for different predictions
if max_class == "CONFIRMED":
    st.success(f"✅ **Prediction: {label}** (Confidence: {confidence:.1%})")
    st.write("🎉 **Strong exoplanet signal detected!** This target shows characteristics consistent with confirmed exoplanets.")
elif max_class == "CANDIDATE": 
    st.info(f"🔍 **Prediction: {label}** (Confidence: {confidence:.1%})")
    st.write("🤔 **Potential exoplanet candidate.** Requires further investigation to rule out false positives.")
else:  # FALSE POSITIVE
    st.warning(f"❌ **Prediction: {label}** (Confidence: {confidence:.1%})")
    st.write("⚠️ **Likely false positive.** Transit-like signal may be caused by stellar variability, binary eclipses, or instrumental artifacts.")

# Show validation warnings if any
if 'validation_warnings' in st.session_state and st.session_state['validation_warnings']:
    st.subheader("⚠️ Validation Warnings")
    st.write("The following issues were detected that increase the likelihood of a false positive:")
    for warning in st.session_state['validation_warnings']:
        st.write(f"• {warning}")
    
    if max_class != "FALSE POSITIVE":
        st.info("💡 **Note**: Despite these warnings, the model still predicts this as a potential transit. Consider additional follow-up observations.")

# Add navigation to Explainable AI
if confidence > 0.5:  # Only show if we have reasonable confidence
    st.info("🔬 **Want to understand why?** Visit the Explainable AI page to see which features influenced this prediction.")
    if st.button("🔬 Explain This Prediction", type="secondary"):
        st.switch_page("pages/3_🔬_Explainable_AI.py")

# Probability breakdown
if probs:
    st.subheader("📈 Class Probabilities")
    prob_df = pd.DataFrame.from_dict(probs, orient='index', columns=['Probability'])
    prob_df = prob_df.sort_values('Probability', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(prob_df.index, prob_df['Probability'])
    
    # Color bars by class
    colors = {'CONFIRMED': 'green', 'CANDIDATE': 'orange', 'FALSE POSITIVE': 'red'}
    for bar, class_name in zip(bars, prob_df.index):
        bar.set_color(colors.get(class_name, 'blue'))
    
    ax.set_xlabel('Probability')
    ax.set_title('Classification Probabilities')
    ax.set_xlim(0, 1)
    
    # Add percentage labels
    for i, (class_name, prob) in enumerate(prob_df['Probability'].items()):
        ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
    
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# Model information
with st.expander("🔧 Model Details"):
    st.write("**Model Type**: RandomForest Classifier")
    st.write("**Features Used**:", feature_cols if feature_cols else "Unknown")
    st.write("**Training Data**: NASA KOI (Kepler Objects of Interest) dispositions")
    
    if feature_stats:
        st.write("**Feature Statistics** (from training data):")
        st.json(feature_stats)

# Interpretation guidelines
st.subheader("📚 Interpretation Guide")

interpretation = {
    "CONFIRMED": {
        "meaning": "High confidence this is a real exoplanet",
        "action": "✅ Likely a genuine planetary transit",
        "icon": "✅"
    },
    "CANDIDATE": {
        "meaning": "Promising signal that needs further validation", 
        "action": "🔍 Requires additional observations for confirmation",
        "icon": "🔍"
    },
    "FALSE POSITIVE": {
        "meaning": "Likely not a planetary transit",
        "action": "❌ Probably stellar activity, binary star, or instrument artifact",
        "icon": "❌"
    }
}

if label in interpretation:
    info = interpretation[label]
    st.info(f"""
    **{info['icon']} {label}**
    
    **Meaning**: {info['meaning']}
    
    **Recommended Action**: {info['action']}
    """)

# Confidence assessment
if confidence > 0.8:
    st.success("🎯 **High Confidence**: The model is very confident in this prediction.")
elif confidence > 0.6:
    st.info("👍 **Moderate Confidence**: The prediction is reasonably reliable.")
else:
    st.warning("⚠️ **Low Confidence**: Results should be interpreted with caution.")

# Next steps
st.subheader("🚀 Next Steps")

next_steps = []
if confidence > 0.6:
    next_steps.append("🔬 **Explainability**: Visit the Explainability page to understand why the model made this prediction")
else:
    next_steps.append("🔍 **Re-examine Data**: Consider checking data quality or trying different targets")

if label == "CANDIDATE":
    next_steps.append("📊 **Further Analysis**: Consider additional observations or different analysis methods")

if label == "CONFIRMED":
    next_steps.append("🎉 **Success**: This appears to be a genuine exoplanet detection!")

for step in next_steps:
    st.write(f"- {step}")

# Store prediction results for explainability page
st.session_state['ml_prediction'] = {
    'label': label,
    'probabilities': probs,
    'confidence': confidence,
    'features': X_pred
}