import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Streamlit
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

st.set_page_config(page_title="Explainable AI", page_icon="🔬", layout="wide")

# Import path setup
# Ensure the project root is in the path for shared_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SHAP imports with fallback
try:
    import shap
    _HAS_SHAP = True
except ImportError as e:
    shap = None
    _HAS_SHAP = False
    # Now we can safely use sys here
    st.session_state['shap_import_error'] = {
        "error": str(e),
        "executable": sys.executable,
        "path": sys.path
    }

from shared_utils import summarize_lightcurve, professional_bls

@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load ML model and metadata, training if necessary."""
    model = None
    feature_cols = []
    feature_stats = {}
    importances = {}
    
    models_dir = Path('models')
    model_path = models_dir/'model.joblib'
    
    # Check if model exists, if not train a lightweight version
    if not model_path.exists():
        st.warning("Model not found. Training lightweight model for demonstration...")
        try:
            # Import and run the lightweight trainer
            from utils.model_trainer import train_lightweight_model
            model, feature_cols, feature_stats, importances = train_lightweight_model()
            st.success("Model trained successfully!")
            return model, feature_cols, feature_stats, importances
        except Exception as e:
            st.error(f"Failed to train model: {str(e)}")
            return None, [], {}, {}
    
    # Load existing model
    try:
        import joblib  # type: ignore
        if model_path.exists():
            model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
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
    if not _HAS_SHAP or model is None or X is None:
        return None, None
    
    # Check if X is empty (handle both DataFrame and list/array cases)
    if hasattr(X, 'empty') and X.empty:
        return None, None
    elif hasattr(X, '__len__') and len(X) == 0:
        return None, None
        
    explainer = _get_shap_explainer(model)
    if explainer is None:
        return None, None
        
    try:
        # Ensure X is properly formatted and single sample
        if not isinstance(X, pd.DataFrame):
            return None, None
            
        # Always use only the first row to ensure single sample
        X_single = X.iloc[[0]]  # Keep as DataFrame with single row
        X_values = X.iloc[0]    # Series for feature values
        
        shap_values = explainer.shap_values(X_single)
        proba = model.predict_proba(X_single)[0]
        predicted_class = int(np.argmax(proba))
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Multi-class case - use the predicted class
            if predicted_class < len(shap_values):
                shap_vals = shap_values[predicted_class]
            else:
                shap_vals = shap_values[0]
        else:
            # Binary case or single array
            shap_vals = shap_values
            
        # Ensure we have the right shape for a single sample
        shap_vals = np.asarray(shap_vals)  # Ensure it's a numpy array
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]  # Take first sample
        shap_vals = shap_vals.flatten()  # Ensure 1D
        
        # Force plot - try different approaches for compatibility
        expected_value = explainer.expected_value
        
        # Handle expected_value safely
        if isinstance(expected_value, (list, np.ndarray)):
            expected_len = len(expected_value) if hasattr(expected_value, '__len__') else 1
            if expected_len > predicted_class:
                exp_val = float(expected_value[predicted_class])
            else:
                exp_val = float(expected_value[0]) if expected_len > 0 else 0.0
        else:
            exp_val = float(expected_value)
        
        try:
            # FINAL, MOST ROBUST APPROACH for Streamlit Cloud:
            # The key is to let SHAP create the plot and then immediately
            # pass the figure object to Streamlit without any other
            # matplotlib commands that could affect the global state.
            
            plt.close('all') # Ensure a clean slate before plotting.

            exp_val_scalar = float(exp_val)
            shap_vals_array = np.asarray(shap_vals, dtype=float)
            
            # Generate the plot. `matplotlib=True` makes it create a new figure.
            shap.force_plot(
                exp_val_scalar,
                shap_vals_array,
                X_values,
                matplotlib=True,
                show=False,
                figsize=(12, 3)
            )
            
            # Grab the current figure that SHAP just created.
            fig = plt.gcf()
            fig.tight_layout()

        except Exception as e:
            # Fallback approach - create custom bar plot
            print(f"SHAP error: {e}")  # Debug info
            fig, ax = plt.subplots(figsize=(12, 3))
            feature_names = X.columns.tolist()
            shap_vals_safe = np.asarray(shap_vals, dtype=float).flatten()
            colors = ['red' if val > 0 else 'blue' for val in shap_vals_safe]
            ax.bar(range(len(shap_vals_safe)), shap_vals_safe, color=colors, alpha=0.7)
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_ylabel('SHAP Value')
            ax.set_title(f'Feature Contributions (Base: {float(exp_val):.3f})')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            fig.tight_layout()
        
        return fig, shap_vals
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
    st.write("SHAP (SHapley Additive exPlanations) is not installed or accessible in this environment.")
    
    if 'shap_import_error' in st.session_state:
        with st.expander("Technical Details"):
            st.write("Here is the detailed error information:")
            st.json(st.session_state['shap_import_error'])
    else:
        st.code("pip install shap", language="bash")
    
    st.info("SHAP provides detailed explanations of individual predictions by showing how each feature contributes.")
else:
    st.write("SHAP shows **how each feature contributed** to this specific prediction:")
    
    with st.spinner("🧠 Computing SHAP explanations..."):
        shap_fig, shap_values = get_shap_plot(model, X_features)
    
    if shap_fig is not None:
        # Using clear_figure=True is often necessary on Streamlit Cloud
        # to ensure proper resource management, even if it seems
        # counter-intuitive. The key is the robust plot creation above.
        st.pyplot(shap_fig, clear_figure=True)
        
        # Explain the SHAP plot
        st.write("**How to read this plot:**")
        st.write("- 🔴 **Red**: Features pushing prediction toward this class")
        st.write("- 🔵 **Blue**: Features pushing prediction away from this class") 
        st.write("- **Arrow length**: Strength of feature contribution")
        st.write("- **Base value**: Model's average prediction")
        
        # Feature contribution analysis
        if shap_values is not None and len(shap_values) > 0 and X_features is not None and not X_features.empty:
            st.subheader("📊 Feature Contribution Analysis")
            
            # Ensure all arrays have the same length with robust checks
            shap_values_flat = np.asarray(shap_values).flatten()
            feature_values = X_features.iloc[0].values
            
            num_features = len(feature_cols)
            
            # Check if arrays match the expected number of features
            if len(shap_values_flat) == num_features and len(feature_values) == num_features:
                contrib_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Value': feature_values,
                    'SHAP_Value': shap_values_flat,
                    'Contribution': ['Positive' if sv > 0 else 'Negative' for sv in shap_values_flat]
                })
                contrib_df['Abs_SHAP'] = np.abs(contrib_df['SHAP_Value'])
                contrib_df = contrib_df.sort_values('Abs_SHAP', ascending=False)
                
                # Show top contributing features
                st.write("**Top Contributing Features:**")
                for i, row in contrib_df.head(5).iterrows():
                    direction = "↗️" if row['SHAP_Value'] > 0 else "↘️"
                    st.write(f"{direction} **{row['Feature']}**: {row['SHAP_Value']:+.3f} (value: {row['Value']:.3g})")
            else:
                st.warning("⚠️ Could not generate feature contribution table due to mismatched data lengths.")
                st.caption(f"Debug info: Features({len(feature_cols)}), Values({len(feature_values)}), SHAP({len(shap_values_flat)})")
            
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

if _HAS_SHAP and shap_values is not None and len(shap_values) > 0:
    # Ensure shap_values and feature_cols have the same length before processing
    if len(shap_values) == len(feature_cols):
        # Find most influential feature
        max_contrib_idx = np.argmax(np.abs(shap_values))
        max_feature = feature_cols[max_contrib_idx]
        max_contribution = shap_values[max_contrib_idx]
        
        if max_contribution > 0:
            insights.append(f"🔍 **{max_feature}** was the strongest factor supporting the '{prediction_label}' classification.")
        else:
            insights.append(f"🔍 **{max_feature}** was the strongest factor working against the '{prediction_label}' classification.")
    else:
        insights.append("⚠️ Could not determine the most influential feature due to a data mismatch.")

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