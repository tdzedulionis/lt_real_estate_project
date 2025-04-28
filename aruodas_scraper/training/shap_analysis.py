"""Module for SHAP (SHapley Additive exPlanations) analysis of models."""

import numpy as np
import matplotlib.pyplot as plt
import shap
import os

def analyze_with_shap(model, X_processed, feature_names, shap_dir, model_name="model"):
    """Generate and visualize SHAP explanations for model feature importance.
    
    Creates summary and importance plots, saves them to disk, and returns SHAP values 
    with feature importance metrics."""
    print(f"Starting SHAP analysis for {model_name}...")
    
    # Create SHAP explainer based on model type
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    
    # Calculate SHAP values (use a smaller sample if data is large)
    sample_size = min(500, X_processed.shape[0])
    if X_processed.shape[0] > sample_size:
        print(f"Using {sample_size} samples for SHAP analysis (full dataset: {X_processed.shape[0]} rows)")
        sample_indices = np.random.choice(X_processed.shape[0], sample_size, replace=False)
        X_sample = X_processed[sample_indices]
    else:
        X_sample = X_processed
    
    # Get SHAP values
    shap_values = explainer(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    values_matrix = shap_values.values if hasattr(shap_values, 'values') else shap_values
    shap_importance = np.abs(values_matrix).mean(axis=0)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f'shap_summary_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    # Importance plot
    plt.figure(figsize=(12, 8))
    shap_importance_order = np.argsort(-shap_importance)
    plt.barh(range(len(shap_importance_order[:20])), 
             shap_importance[shap_importance_order[:20]])
    plt.yticks(range(len(shap_importance_order[:20])), 
               [feature_names[i] for i in shap_importance_order[:20]])
    plt.title(f"SHAP Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f'shap_importance_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    # Print top features
    top_features = [(feature_names[i], shap_importance[i]) for i in shap_importance_order[:10]]
    print(f"Top 10 features by SHAP importance for {model_name}:")
    for i, (feature, importance) in enumerate(top_features):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    return shap_values, shap_importance, top_features
