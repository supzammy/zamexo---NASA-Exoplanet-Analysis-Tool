# ZAMEXO demo outline (2–3 minutes)

- Problem: Detecting and classifying exoplanet transits from NASA missions.
- Data: MAST light curves (Kepler/K2/TESS) + Exoplanet Archive (KOI/TOI/K2).
- Method:
  - BLS to find transit signals; features: period, duration, depth, SDE.
  - RandomForest baseline trained on KOI; optional multi-source model.
  - Explainability via feature importances, optional SHAP.
- Demo:
  - Search target (e.g., Kepler-10), show light curve and phase-folded plot.
  - AI panel: prediction + probabilities.
  - Upload CSV, run BLS, classify.
- Impact: Fast triage of candidate signals; future: better models, UX polish.
