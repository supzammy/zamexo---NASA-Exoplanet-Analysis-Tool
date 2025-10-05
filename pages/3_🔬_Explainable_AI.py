import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

st.set_page_config(page_title="Explainable AI", page_icon="🔬", layout="wide")

# SHAP imports with fallback
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

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

@st.cache_resource(show_spinner=False)
def _get_shap_explainer(_model):
    """Get SHAP explainer (cached)."""
    if not _HAS_SHAP or _model is None:
        return None
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

def get_shap_plot(model, X):
    """Generate SHAP force plot."""
    if not _HAS_SHAP or model is None or X is None or X.empty:
        return None, None
        
    explainer = _get_shap_explainer(model)
    if explainer is None:
        return None, None
        
    try:
        shap_values = explainer.shap_values(X)
        proba = model.predict_proba(X)[0]
        predicted_class = int(np.argmax(proba))
        
        # Force plot
        fig, ax = plt.subplots(figsize=(12, 3))
        shap.force_plot(
            explainer.expected_value[predicted_class],
            shap_values[predicted_class][0],  # First sample
            X.iloc[0],
            matplotlib=True,
            show=False,
            ax=ax
        )
        fig.tight_layout()
        
        return fig, shap_values[predicted_class][0]
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        return None, None

def plot_feature_importance(importances, feature_cols):
    """Plot global feature importance."""
    if not importances or not feature_cols:
        st.info("No feature importance data available.")
        return
        
    # Convert to series and sort
    imp_series = pd.Series(importances, index=feature_cols)
    imp_series = imp_series.sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp_series) * 0.4)))
    bars = ax.barh(range(len(imp_series)), imp_series.values)
    ax.set_yticks(range(len(imp_series)))
    ax.set_yticklabels(imp_series.index)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Global Feature Importance (Random Forest)')
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, v in enumerate(imp_series.values):
        ax.text(v + max(imp_series.values) * 0.01, i, f'{v:.3f}', va='center')
    
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.title("🔬 Explainable AI")

st.title("🔬 Explainable AI")

st.write("""
Understand how the machine learning model makes its predictions using SHAP (SHapley Additive exPlanations) 
and feature importance analysis.
""")

# Load model artifacts
model, feature_cols, feature_stats, importances = load_artifacts()

if model is None:
    st.error("❌ **Model not found!**")
    st.write("Please train the model first:")
    st.code("python scripts/train_baseline.py", language="bash")
    st.stop()

# Check for both BLS results and ML predictions
if 'bls_results' not in st.session_state or 'ml_prediction' not in st.session_state:
    st.warning("⚠️ **No prediction results found.**")
    st.write("Please complete the analysis workflow first:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Step 1**: Generate transit features")
        if st.button("🔍 Go to Transit Detection", type="primary", use_container_width=True):
            st.switch_page("pages/1_🔍_Transit_Detection.py")
    
    with col2:
        st.write("**Step 2**: Get ML predictions")
        if st.button("🤖 Go to AI Classification", type="primary", use_container_width=True):
            st.switch_page("pages/2_🤖_AI_Classification.py")
    
    st.info("💡 **Tip**: The explainability analysis requires both transit features and classification results.")
    st.stop()

# Get data from session state
bls = st.session_state['bls_results']
ml_pred = st.session_state['ml_prediction']
source_label = st.session_state.get('source_label', 'Unknown')

# Display current prediction context
st.subheader("📊 Current Analysis")
col1, col2, col3 = st.columns(3)

prediction_label = ml_pred['label']
probabilities = ml_pred['probabilities']
max_prob = max(probabilities.values()) if probabilities else 0.0
X_features = ml_pred.get('features', None)

col1.metric("🎯 Prediction", prediction_label)
col2.metric("🔢 Confidence", f"{max_prob:.1%}")
col3.metric("📡 Source", source_label)

# Color-code confidence
if max_prob > 0.8:
    st.success("🎯 **High confidence prediction**")
elif max_prob > 0.6:
    st.info("🤔 **Moderate confidence prediction**")
else:
    st.warning("⚠️ **Low confidence prediction**")

# Global Feature Importance
st.subheader("🌍 Global Feature Importance")
st.write("These are the features the model considers most important **across all predictions**:")

plot_feature_importance(importances, feature_cols)

if importances and feature_cols:
    # Top features explanation
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
    st.write("**Top 3 Most Important Features:**")
    for i, (feature, importance) in enumerate(top_features, 1):
        st.write(f"{i}. **{feature}**: {importance:.3f}")

# SHAP Analysis
st.subheader("🔍 Individual Prediction Explanation (SHAP)")

if not _HAS_SHAP:
    st.error("❌ **SHAP not available**")
    st.write("SHAP (SHapley Additive exPlanations) is not installed in this environment.")
    st.code("pip install shap", language="bash")
    st.info("SHAP provides detailed explanations of individual predictions by showing how each feature contributes.")
else:
    st.write("SHAP shows **how each feature contributed** to this specific prediction:")
    
    with st.spinner("🧠 Computing SHAP explanations..."):
        shap_fig, shap_values = get_shap_plot(model, X_features)
    
    if shap_fig is not None:
        st.pyplot(shap_fig, clear_figure=True)
        
        # Explain the SHAP plot
        st.write("**How to read this plot:**")
        st.write("- 🔴 **Red**: Features pushing prediction toward this class")
        st.write("- 🔵 **Blue**: Features pushing prediction away from this class") 
        st.write("- **Arrow length**: Strength of feature contribution")
        st.write("- **Base value**: Model's average prediction")
        
        # Feature contribution analysis
        if shap_values is not None and len(shap_values) > 0:
            st.subheader("📊 Feature Contribution Analysis")
            
            # Create feature contribution dataframe
            contrib_df = pd.DataFrame({
                'Feature': feature_cols,
                'Value': X_features.iloc[0].values,
                'SHAP_Value': shap_values,
                'Contribution': ['Positive' if sv > 0 else 'Negative' for sv in shap_values]
            })
            contrib_df['Abs_SHAP'] = np.abs(contrib_df['SHAP_Value'])
            contrib_df = contrib_df.sort_values('Abs_SHAP', ascending=False)
            
            # Show top contributing features
            st.write("**Top Contributing Features:**")
            for i, row in contrib_df.head(5).iterrows():
                direction = "↗️" if row['SHAP_Value'] > 0 else "↘️"
                st.write(f"{direction} **{row['Feature']}**: {row['SHAP_Value']:+.3f} (value: {row['Value']:.3g})")
            
            # Feature values vs training statistics
            if feature_stats:
                st.subheader("📈 Feature Values vs Training Data")
                st.write("How do this example's features compare to the training data?")
                
                comparison_data = []
                for feature in feature_cols:
                    if feature in feature_stats.get('mean', {}):
                        actual_val = X_features.iloc[0][feature]
                        mean_val = feature_stats['mean'][feature]
                        std_val = feature_stats.get('std', {}).get(feature, 1.0)
                        
                        z_score = (actual_val - mean_val) / std_val if std_val > 0 else 0
                        
                        comparison_data.append({
                            'Feature': feature,
                            'This Example': actual_val,
                            'Training Mean': mean_val,
                            'Z-Score': z_score,
                            'Interpretation': 'Above average' if z_score > 0.5 else 'Below average' if z_score < -0.5 else 'Typical'
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    st.dataframe(comp_df, use_container_width=True)
    else:
        st.error("❌ Failed to generate SHAP explanation.")
        st.write("This might be due to:")
        st.write("- Incompatible feature format")
        st.write("- Model complexity")
        st.write("- SHAP version issues")

# Model Confidence Analysis
st.subheader("🎯 Confidence Analysis")

if probabilities:
    st.write("**Class Probability Breakdown:**")
    prob_df = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probability'])
    prob_df['Percentage'] = prob_df['Probability'] * 100
    st.dataframe(prob_df, use_container_width=True)
    
    # Confidence interpretation
    max_prob = max(probabilities.values())
    second_max = sorted(probabilities.values(), reverse=True)[1] if len(probabilities) > 1 else 0
    margin = max_prob - second_max
    
    st.write(f"**Decision Margin**: {margin:.1%}")
    if margin > 0.4:
        st.success("🎯 **Clear Decision**: Large margin between top classes indicates high confidence.")
    elif margin > 0.2:
        st.info("👍 **Moderate Decision**: Reasonable separation between classes.")
    else:
        st.warning("⚠️ **Uncertain Decision**: Small margin suggests the model is less certain.")

# Actionable Insights
st.subheader("💡 Actionable Insights")

insights = []

if _HAS_SHAP and shap_values is not None:
    # Find most influential feature
    max_contrib_idx = np.argmax(np.abs(shap_values))
    max_feature = feature_cols[max_contrib_idx]
    max_contribution = shap_values[max_contrib_idx]
    
    if max_contribution > 0:
        insights.append(f"🔍 **{max_feature}** was the strongest factor supporting the '{prediction_label}' classification.")
    else:
        insights.append(f"🔍 **{max_feature}** was the strongest factor working against the '{prediction_label}' classification.")

if max_prob < 0.6:
    insights.append("⚠️ **Low confidence** suggests this target may be borderline. Consider additional observations.")

if prediction_label == "FALSE POSITIVE":
    insights.append("❌ The model thinks this is likely **not a real planet**. This could be stellar variability or instrumental effects.")
elif prediction_label == "CANDIDATE":
    insights.append("🔍 The model sees **promising signals** but isn't fully confident. Further validation recommended.")
elif prediction_label == "CONFIRMED":
    insights.append("✅ The model is confident this represents a **real planetary transit**!")

for insight in insights:
    st.write(f"- {insight}")

# Technical Details
with st.expander("🔧 Technical Details"):
    st.write("**Model Architecture**: RandomForest Classifier")
    st.write("**Explainability Method**: SHAP (SHapley Additive exPlanations)")
    st.write("**Feature Engineering**: Box Least Squares (BLS) transit detection")
    st.write("**Training Data**: NASA Kepler Objects of Interest (KOI) catalog")
    
    if _HAS_SHAP:
        st.write("**SHAP Method**: TreeExplainer (optimized for tree-based models)")
    
    st.write("**Feature Set**:", feature_cols if feature_cols else "Unknown")