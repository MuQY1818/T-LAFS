import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_permutation_importance(model, X_val, y_val, metric_func, higher_is_better=True, n_repeats=1):
    """
    Calculates permutation importance for a trained model.

    Args:
        model: A trained model with a `predict` method.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series or np.array): Validation target.
        metric_func (callable): A function to calculate the performance metric (e.g., r2_score).
        higher_is_better (bool): Whether a higher metric value indicates better performance.
        n_repeats (int): The number of times to repeat the permutation for each feature to get a more stable result.

    Returns:
        pd.DataFrame: A DataFrame with features and their importance scores and standard deviations.
    """
    logging.info("Calculating permutation importance...")
    
    # 1. Calculate baseline performance
    baseline_preds = model.predict(X_val)
    baseline_performance = metric_func(y_val, baseline_preds)
    logging.info(f"Baseline performance ({metric_func.__name__}): {baseline_performance:.4f}")

    importances = {}
    
    # Using tqdm for progress bar
    for col in tqdm(X_val.columns, desc="Permuting features"):
        performance_drops = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            # Shuffle the current column
            np.random.shuffle(X_permuted[col].values)
            
            # 2. Make predictions on permuted data
            permuted_preds = model.predict(X_permuted)
            
            # 3. Calculate permuted performance
            permuted_performance = metric_func(y_val, permuted_preds)
            
            # 4. Calculate performance drop
            if higher_is_better:
                drop = baseline_performance - permuted_performance
            else:
                drop = permuted_performance - baseline_performance
            performance_drops.append(drop)
        
        importances[col] = {
            'importance_mean': np.mean(performance_drops),
            'importance_std': np.std(performance_drops)
        }

    # Convert to DataFrame for easier analysis
    importance_df = pd.DataFrame.from_dict(importances, orient='index')
    importance_df = importance_df.sort_values(by='importance_mean', ascending=False)
    
    logging.info("Permutation importance calculation finished.")
    return importance_df

def analyze_iterative_contribution(history):
    """
    Analyzes the iterative contribution of features from the T-LAFS history.

    Args:
        history (list of dicts): The history log from the TLAFS_Algorithm instance.

    Returns:
        pd.DataFrame: A DataFrame detailing the marginal contribution of each feature.
    """
    logging.info("Analyzing iterative feature contribution...")
    if not history or len(history) < 2:
        logging.warning("History is too short to analyze iterative contribution.")
        return pd.DataFrame()

    contributions = []
    # The first entry in history is the baseline
    previous_performance = history[0].get('performance', 0)

    for i in range(1, len(history)):
        item = history[i]
        current_performance = item.get('performance', 0)
        added_feature = item.get('feature_name', 'N/A')
        
        marginal_contribution = current_performance - previous_performance
        
        contributions.append({
            'step': i,
            'feature_name': added_feature,
            'marginal_contribution': marginal_contribution,
            'cumulative_performance': current_performance,
        })
        
        previous_performance = current_performance

    contribution_df = pd.DataFrame(contributions)
    contribution_df = contribution_df.sort_values(by='marginal_contribution', ascending=False)

    logging.info("Iterative contribution analysis finished.")
    return contribution_df 