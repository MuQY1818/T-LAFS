"""
This script runs a standalone experiment for the 'kitchen sink' control group.
It loads the data, applies the massive feature generation macro, and evaluates
the performance of the resulting feature set without running the T-LAFS loop.
"""
import pandas as pd
import os
import sys

# Add the project root to the Python path to allow for absolute imports
# This is necessary for running the script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tlafs.features.generator import execute_plan
from tlafs.utils.evaluation import probe_feature_set
from tlafs.utils.data_utils import get_time_series_data

def run_control_group_experiment():
    """
    Main function to run the kitchen sink control experiment.
    """
    print("🚀 Starting Control Group Experiment: 'Kitchen Sink' Feature Set")
    
    # --- 1. Load Data ---
    # Same data loading logic as in sonata.py to ensure consistency
    df = get_time_series_data('min_daily_temps')
    target_col = 'Target' # Use the standardized column name
    
    # Ensure consistent column names
    df = df.rename(columns={'Date': 'date'})
    
    # Add this line to ensure the second column is named 'Target'
    df.columns = ['date', 'Target']
    
    print(f"✅ Data loaded successfully. Shape: {df.shape}")

    # --- 2. Define and Execute the Plan ---
    kitchen_sink_plan = [{
        "function": "create_kitchen_sink_features_macro",
        "args": {"col": target_col}
    }]
    
    # The `tlafs_params` dictionary can be empty as this macro is self-contained
    tlafs_params = {
        "target_col_static": target_col
    }
    
    print("\nApplying the 'create_kitchen_sink_features_macro'...")
    df_kitchen_sink, feature_name = execute_plan(df.copy(), kitchen_sink_plan, tlafs_params)
    
    if df_kitchen_sink is None:
        print("❌ Fatal: Failed to generate features from the kitchen sink macro.")
        return
    
    print(f"✅ Macro executed. New feature set name: '{feature_name}'")

    # 输出特征数量和特征名
    features_to_probe = [c for c in df_kitchen_sink.columns if c not in ['date', target_col]]
    print(f"Number of kitchen sink features: {len(features_to_probe)}")
    print(f"Feature names: {features_to_probe}")
    
    # --- 3. Evaluate the Performance ---
    print("\nEvaluating the performance of the generated feature set...")
    features_to_probe = [c for c in df_kitchen_sink.columns if c not in ['date', target_col]]
    
    # The probe function handles splitting, training, and evaluating
    kitchen_sink_score, _, _ = probe_feature_set(df_kitchen_sink, target_col, features_to_probe)
    
    print("\n" + "="*50)
    print("🏆 Control Group Experiment Finished 🏆")
    print(f"Final R² Score for 'Kitchen Sink' features: {kitchen_sink_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_control_group_experiment() 