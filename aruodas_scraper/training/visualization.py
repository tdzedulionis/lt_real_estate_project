"""Module for creating model performance visualizations."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, cv=5, train_sizes=np.linspace(.1, 1.0, 5), 
                        title="Learning Curves", save_path=None):
    """Plot training and cross-validation learning curves to analyze model performance and overfitting."""
    if save_path is None:
        return
        
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='r2')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("R² Score")
    plt.legend(loc="best")
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Create bar plot of top feature importances from tree-based models, returns importance dictionary."""
    if not save_path or not hasattr(model, 'feature_importances_'):
        return None
        
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return dict(zip([feature_names[i] for i in indices], importances[indices]))

def plot_predictions_by_feature(X_test, y_test, y_pred, feature_name, save_path=None):
    """Create scatter plot comparing actual vs predicted values against a specific feature."""
    if save_path is None:
        return
        
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test[feature_name], y_test, alpha=0.6, color='blue', label='Actual')
    plt.scatter(X_test[feature_name], y_pred, alpha=0.6, color='red', label='Predicted')
    plt.xlabel(feature_name)
    plt.ylabel('Price')
    plt.title(f'Price vs. {feature_name}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_test, predictions, save_path=None):
    """Create scatter plot of residuals vs predicted values to analyze prediction errors."""
    if save_path is None:
        return
        
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(model_results, final_metrics, save_path=None):
    """Create bar plot comparing RMSE and R² scores across different models including ensemble."""
    if save_path is None:
        return
        
    plt.figure(figsize=(12, 6))
    model_names = list(model_results.keys()) + ['Ensemble']
    rmse_values = [model_results[name]['RMSE'] for name in model_results.keys()] + [final_metrics[0]]
    r2_values = [model_results[name]['R2'] for name in model_results.keys()] + [final_metrics[2]]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, rmse_values, width, label='RMSE')
    plt.bar(x + width/2, r2_values, width, label='R²')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
