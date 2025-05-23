#!/usr/bin/env python3
"""Module for training and evaluating machine learning models."""

import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import lightgbm as lgbm
import joblib
import os
from aruodas_scraper.config.settings import PROPERTY_TYPES
from aruodas_scraper.training.visualization import (plot_learning_curves, plot_feature_importance, 
                          plot_residuals, plot_model_comparison)
from aruodas_scraper.training.utils import get_property_config, create_model_directories, prepare_features
import warnings
# Filter out specific warning types
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*No further splits.*')
warnings.filterwarnings('ignore', message='.*Base score.*')
warnings.filterwarnings('ignore', message='.*label encoder.*')
warnings.filterwarnings('ignore', message='.*DataConversionWarning.*')
warnings.filterwarnings('ignore', message='.*Optimization.*did not converge.*')

def evaluate_model(model_name, y_true, y_pred):
    """Evaluate model performance using multiple metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.4f}, R²: {r2:.4f}')
    return rmse, mae, r2, mape

def final_ml_analysis(df, target='price', test_size=0.2,
                      random_state=42, save_model=True, output_dir=None, log = False):
    """Perform complete machine learning analysis including model training and evaluation."""
    if output_dir is None:
        raise ValueError("output_dir must be provided")
        
    print("="*80)
    print("Final ML Analysis for Housing Price Prediction")
    print("="*80)
    
    # Get property config and create directories
    property_config = get_property_config(output_dir, PROPERTY_TYPES)
    create_model_directories(property_config)
    
    # Data preparation
    df, numeric_features, categorical_features = prepare_features(df, target)
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Preprocessing Pipeline
    print("\n1. Setting up Data Preprocessing Pipeline")
    numeric_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', PowerTransformer(method='yeo-johnson'))
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Data Splitting
    print("\n2. Splitting data into train and test sets")
    X = df.drop(target, axis=1)
    print("Applying log transform to target variable")
    if log == True:
        y = np.log1p(df[target])
    else:
        y = df[target]
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Model Setup
    print("\n3. Setting up models and hyperparameter grids")
    models_and_params = {
        'LightGBM': (
            lgbm.LGBMRegressor(
                random_state=random_state,
                verbose=-1,
                min_child_samples=20,
                min_split_gain=0.1
            ),
            {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__num_leaves': [31, 63, 127],
                'model__reg_alpha': [0, 0.1, 0.5]
            }
        ),
        'XGBoost': (
            XGBRegressor(
                random_state=random_state,
                verbosity=0,
                min_child_weight=5,
                gamma=0.1
            ),
            {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__min_child_weight': [1, 3, 5]
            }
        ),
        'Gradient Boosting': (
            GradientBoostingRegressor(
                random_state=random_state,
                min_samples_split=20,
                min_impurity_decrease=0.1
            ),
            {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 1.0]
            }
        )
    }
    
    # Train and evaluate models
    print("\n4. Training and evaluating models")
    results = {}
    trained_models = {}
    feature_names = None
    
    for name, (model, params) in models_and_params.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=params,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        
        y_pred = best_pipeline.predict(X_test)
        rmse, mae, r2, mape = evaluate_model(name, y_test, y_pred)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
        trained_models[name] = best_pipeline
        
        if feature_names is None and hasattr(best_pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Cross-validation
    print("\n5. Performing cross-validation")
    cv_results = {}
    for name, pipeline in trained_models.items():
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
        cv_results[name] = {'scores': cv_scores, 'mean': cv_scores.mean(), 'std': cv_scores.std()}
    
    # Meta-model training
    meta_features_train = np.column_stack([
        model.predict(X_train) for model in trained_models.values()
    ])
    meta_features_test = np.column_stack([
        model.predict(X_test) for model in trained_models.values()
    ])
    
    meta_grid_search = GridSearchCV(
        Ridge(),
        param_grid={
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky']
        },
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    meta_grid_search.fit(meta_features_train, y_train)
    meta_model = meta_grid_search.best_estimator_
    
    # Final evaluation
    final_predictions = meta_model.predict(meta_features_test)
    final_metrics = evaluate_model('Ensemble', y_test, final_predictions)

    # Generate visualizations
    for name, model in trained_models.items():
        # Learning curves
        plot_learning_curves(
            model, X, y, 
            title=f"Learning Curves - {name}",
            save_path=os.path.join(property_config['viz_dir'], f"learning_curve_{name}.png")
        )
        
        # Feature importance for tree-based models
        if feature_names is not None:
            plot_feature_importance(
                model.named_steps['model'],
                feature_names,
                save_path=os.path.join(property_config['viz_dir'], f"feature_importance_{name}.png")
            )

    # Model performance visualizations
    plot_residuals(y_test, final_predictions, 
                  save_path=os.path.join(property_config['viz_dir'], 'residuals.png'))
    plot_model_comparison(results, final_metrics, 
                         save_path=os.path.join(property_config['viz_dir'], 'model_comparison.png'))

    # SHAP analysis for best model
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = trained_models[best_model_name]
    X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)
    
    # Save models
    if save_model:
        try:
            model_dir = property_config['model_dir']
            if not os.path.exists(model_dir):
                print(f"Creating model directory: {model_dir}")
                os.makedirs(model_dir, exist_ok=True)

            print("\n6. Saving trained models...")
            # Save individual models directly in the models directory
            for name, model in trained_models.items():
                model_path = os.path.join(model_dir, f'{name.lower().replace(" ", "_")}.pkl')
                print(f"Saving {name} model to {model_path}...")
                joblib.dump(model, model_path, protocol=4)
                
                # Verify saved model can be loaded
                try:
                    joblib.load(model_path)
                    print(f"Successfully saved and verified {name} model")
                except Exception as e:
                    raise IOError(f"Failed to verify {name} model after saving: {str(e)}")
            
            # Save ensemble
            ensemble = {
                'base_models': trained_models,
                'meta_model': meta_model,
                'preprocessor': preprocessor,
                'feature_names': feature_names,
                'target_info': {'log_transform': True if log else False}
            }
            ensemble_path = os.path.join(model_dir, 'ensemble.pkl')
            print(f"Saving ensemble model to {ensemble_path}...")
            joblib.dump(ensemble, ensemble_path, protocol=4)
            
            # Verify ensemble can be loaded
            try:
                joblib.load(ensemble_path)
                print("Successfully saved and verified ensemble model")
            except Exception as e:
                raise IOError(f"Failed to verify ensemble model after saving: {str(e)}")
            
            print("\nAll models saved and verified successfully!")
            
        except Exception as e:
            print(f"\nError saving models: {str(e)}")
            raise
    
    return ensemble, {'base_models': results, 'cv': cv_results, 'ensemble': final_metrics}

def predict(ensemble, X):
    """Make predictions using the ensemble model."""
    base_models = ensemble['base_models']
    meta_model = ensemble['meta_model']
    
    meta_features = np.column_stack([
        model.predict(X) for model in base_models.values()
    ])
    predictions = meta_model.predict(meta_features)
    
    # Check if log transform was used during training
    if ensemble.get('target_info', {}).get('log_transform', False):
        return np.expm1(predictions)
    else:
        return predictions

def load_model(model_path):
    """Load a saved ensemble model."""
    try:
        ensemble = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return ensemble
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
