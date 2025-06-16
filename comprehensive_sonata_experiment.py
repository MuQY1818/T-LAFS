# 不同的模型以自己为探针的效果
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import warnings
import google.generativeai as genai
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import datetime
import random
from pytorch_tabnet.tab_model import TabNetRegressor
from catboost import CatBoostRegressor
import re
from collections import defaultdict
from sklearn.inspection import permutation_importance
import time
from sklearn.svm import SVR
from sklearn.linear_model import Lasso

# --- 从现有脚本中导入核心逻辑 ---
# 我们假设 clp_probe_experiment.py 中的必要类和函数是可导入的
# 在实际应用中，最好将这些共享的类和函数重构到一个单独的 utils.py 文件中
# FIX: 导入整个模块以正确处理全局变量的作用域问题
import clp_probe_experiment as tlafs_exp

warnings.filterwarnings('ignore')

# --- 新的、更通用的探针函数 ---
def sonata_probe_feature_set(df: pd.DataFrame, target_col: str, model_instance, requires_scaling=False, is_nn=False, random_state=42):
    """
    一个通用的探针函数，可以用任何符合scikit-learn接口或我们自定义的PyTorch模型来评估特征集。
    "Sonata Probe" - 因为每个模型都在演奏自己的乐曲。
    """
    df_feat = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[df_feat.index][target_col]
    X = df_feat

    if X.empty or y.empty or len(X) < 20:
        return {"primary_score": 0.0, "r2": 0.0, "num_features": 0, "model": None, "X_test": None, "y_test": None, "importances": None}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=random_state)

    if len(X_train) < 1 or len(X_test) < 1:
        return {"primary_score": 0.0, "r2": 0.0, "num_features": X.shape[1], "model": None, "X_test": None, "y_test": None, "importances": None}

    importances = None # 默认没有重要性

    if is_nn:
        # --- 神经网络模型的处理流程 ---
        scaler_x = MinMaxScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_test_s = scaler_x.transform(X_test)

        scaler_y = MinMaxScaler()
        y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        
        # 实例化NN模型
        model = model_instance(input_size=X_train.shape[1])
        preds_scaled = tlafs_exp.train_pytorch_model(model, X_train_s, y_train_s, X_test_s)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        # NN模型没有直接的feature_importances_，但可以返回训练好的模型
        final_model = model
        final_X_test = X_test_s # 返回缩放后的测试集
    else:
        # --- Scikit-learn 模型的处理流程 ---
        from sklearn.base import clone
        model = clone(model_instance)
        
        if requires_scaling:
            scaler = MinMaxScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            final_X_test = X_test_s
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            final_X_test = X_test
        
        final_model = model
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X.columns)

    score = r2_score(y_test, preds)
    primary_score = max(0.0, score)

    return {
        "primary_score": primary_score,
        "r2": score,
        "num_features": X.shape[1],
        "model": final_model,
        "X_test": final_X_test,
        "y_test": y_test,
        "importances": importances
    }


# --- 可解释性分析模块 ---

def calculate_permutation_importance(model, X_test, y_test, feature_names, results_dir, model_name, is_nn=False, scaler_y=None):
    """
    计算并可视化排列重要性。
    【已修改】处理NN模型的特殊情况。
    """
    print("  - Calculating Permutation Importance...")
    if model is None or X_test is None or y_test is None:
        print("    - Skipping: model or test data not available.")
        return None
    
    start_time = time.time()

    if is_nn:
        # 对于NN，我们需要定义一个评分函数
        def nn_scorer(model, X, y):
            y_s = scaler_y.transform(y.values.reshape(-1, 1))
            preds_s = tlafs_exp.train_pytorch_model(model, X, y_s, X) # A bit of a hack, but works for scoring
            preds = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).flatten()
            return r2_score(y, preds)

        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=nn_scorer
        )
    else:
        # 对于SKlearn模型，使用默认评分器
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
    elapsed_time = time.time() - start_time
    print(f"    - Permutation importance calculated in {elapsed_time:.2f} seconds.")

    perm_sorted_idx = result.importances_mean.argsort()
    importance_df = pd.DataFrame(
        data={'importance': result.importances_mean[perm_sorted_idx]},
        index=np.array(feature_names)[perm_sorted_idx]
    ).sort_values(by='importance', ascending=False)

    # 可视化
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, max(6, len(perm_sorted_idx) // 3)))
    top_n = 20 
    importance_df.head(top_n).sort_values(by='importance').plot(kind='barh', ax=ax)
    ax.set_title(f'Permutation Importance (Top {top_n})\nModel: {model_name}')
    ax.set_xlabel("Importance Score")
    ax.legend_.remove()
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "permutation_importance.png")
    plt.savefig(plot_path)
    print(f"    - Permutation importance plot saved to {plot_path}")
    plt.close(fig)
    
    return importance_df

def calculate_iterative_contribution_importance(tlafs_history, baseline_score, results_dir, model_name):
    """
    【创新方法】计算并可视化特征的迭代贡献重要性。
    追踪每个特征在被引入时对模型分数的边际贡献。
    """
    print("  - Calculating Iterative Contribution Importance...")
    if not tlafs_history:
        print("    - Skipping: T-LAFS history is empty.")
        return None

    feature_contributions = defaultdict(float)
    current_score = baseline_score
    
    # 我们需要一个临时的DataFrame来确定一个计划会创建哪些列
    dummy_df = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'temp': [10]})

    for record in tlafs_history:
        if record.get('adopted', False):
            plan_extension = record.get('plan', [])
            new_score = record['probe_results']['primary_score']
            score_delta = new_score - current_score

            # 执行计划以找出新创建的列名
            initial_cols = set(dummy_df.columns)
            df_after_plan = tlafs_exp.TLAFS_Algorithm.execute_plan(dummy_df.copy(), plan_extension)
            new_cols = set(df_after_plan.columns) - initial_cols

            if new_cols:
                # 将分数变化平分给这个计划中新创建的所有特征
                contribution_per_feature = score_delta / len(new_cols)
                for col in new_cols:
                    feature_contributions[col] += contribution_per_feature
            
            current_score = new_score # 更新当前分数

    if not feature_contributions:
        print("    - No new features were adopted. Cannot calculate contribution importance.")
        return None

    importance_df = pd.DataFrame(
        feature_contributions.items(), columns=['feature', 'iterative_contribution']
    ).sort_values('iterative_contribution', ascending=False).set_index('feature')

    # 可视化
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) // 3)))
    top_n = 20
    importance_df.head(top_n).sort_values(by='iterative_contribution').plot(kind='barh', ax=ax, color='green')
    ax.set_title(f'Iterative Contribution Importance (Top {top_n})\nModel: {model_name}')
    ax.set_xlabel("Cumulative Score Contribution (ΔR²)")
    ax.legend_.remove()
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "iterative_contribution_importance.png")
    plt.savefig(plot_path)
    print(f"    - Iterative Contribution plot saved to {plot_path}")
    plt.close(fig)

    return importance_df

# --- 修改 TLAFS 算法以接受外部探针 ---
class Sonata_TLAFS_Algorithm(tlafs_exp.TLAFS_Algorithm):
    def __init__(self, base_df, target_col, n_iterations, results_dir, probe_model_instance, model_requires_scaling=False, is_nn=False):
        # 调用父类的构造函数
        super().__init__(base_df, target_col, n_iterations, 0.01, results_dir)
        # 存储奏鸣曲探针模型
        self.probe_model_instance = probe_model_instance
        self.model_requires_scaling = model_requires_scaling
        self.is_nn = is_nn

    def build_llm_context(self, probe_results, iteration_num):
        """
        【已重写】覆盖基类方法，将特征重要性添加到上下文中。
        """
        # 从基类方法开始
        context = super().build_llm_context(probe_results, iteration_num)
        model_name = self.probe_model_instance.__name__ if self.is_nn else self.probe_model_instance.__class__.__name__

        # 动态重命名分数键，使其更具信息性
        if "Current Transformer R2 Score" in context:
             context[f"Current {model_name} R2 Score"] = context.pop("Current Transformer R2 Score")

        # 添加特征重要性（如果可用）
        importances = probe_results.get("importances")
        if importances is not None and not importances.empty:
            sorted_importances = importances.sort_values(ascending=False)
            top_n = 3
            bottom_n = 3
            context["Top Features (High Importance)"] = sorted_importances.head(top_n).to_dict()
            context["Bottom Features (Low Importance)"] = sorted_importances.tail(bottom_n).to_dict()
        
        return context

    def format_prompt_for_llm(self, context_dict, in_context_examples_str):
        """
        【已重写】覆盖基类方法，以格式化包含重要性的新上下文。
        """
        prompt = "--- CURRENT STATE & TASK ---\n"
        prompt += "Your goal is to propose a feature engineering plan to improve the R² score.\n"
        prompt += "Analyze the feature importances below to guide your decisions. Consider deleting low-importance features or creating interactions with high-importance ones.\n"
        
        for key, value in context_dict.items():
            if isinstance(value, (float, np.floating)):
                 prompt += f"- {key}: {value:.4f}\n"
            elif isinstance(value, dict):
                prompt += f"- {key}:\n"
                # 过滤掉值为0的重要性特征，使其更简洁
                filtered_dict = {k: v for k, v in value.items() if v > 0} if "Importance" in key else value
                for k, v in filtered_dict.items():
                    prompt += f"  - {k}: {v:.4f}\n"
            else:
                prompt += f"- {key}: {value}\n"
        
        prompt += "\nPropose a short, creative list of 1-2 operations to improve the score."
        prompt += in_context_examples_str
        return prompt

    def run(self):
        """
        重写 run 方法以使用自定义的 Sonata Probe，并处理需要缩放的模型。
        """
        model_name = self.probe_model_instance.__name__ if self.is_nn else self.probe_model_instance.__class__.__name__

        print(f"\n💡 Starting T-LAFS with {model_name} as the Probe ...\n")
        current_df = self.base_df.copy()
        
        initial_plan = [{"operation": "create_lag", "on": self.target_col, "days": 1, "id": "lag1"}]
        current_df = self.execute_plan(current_df, initial_plan)
        current_plan = initial_plan
        
        print(f"\nEstablishing baseline score with initial feature set (lag1) for {model_name}...")
        try:
            # --- MODIFICATION for scaling & NN ---
            probe_results = sonata_probe_feature_set(current_df, self.target_col, self.probe_model_instance, self.model_requires_scaling, self.is_nn)
            current_score = probe_results["primary_score"]
            self.last_probe_results = probe_results

            self.best_score = current_score
            self.best_df = current_df.copy()
            self.best_plan = current_plan.copy()
            print(f"  - Initial baseline score ({model_name} R²): {self.best_score:.4f} | Features: {probe_results.get('num_features', -1)}")
            
            # Store baseline for iterative contribution calculation
            self.baseline_score_with_lag1 = current_score

        except Exception as e:
            import traceback
            print(f"  - ❌ ERROR during initial evaluation for {model_name}: {e}\n{traceback.format_exc()}")
            return None, None, -1, 0

        for i in range(self.n_iterations):
            print(f"\n----- ITERATION {i+1}/{self.n_iterations} (Probe: {model_name}) -----")
            
            last_results = self.last_probe_results
            print(f"  - Current Score (R²): {last_results['primary_score']:.4f} | #Feats: {last_results.get('num_features', -1)} | Best Score: {self.best_score:.4f}")

            print("\nStep 1: Strategist LLM is devising a new feature combo plan...")
            
            current_state_context = self.build_llm_context(last_results, i)
            in_context_examples = self.experience_buffer.sample(n_good=2, n_bad=1)
            full_prompt = self.format_prompt_for_llm(current_state_context, in_context_examples)
            plan_extension = self.get_plan_from_llm(full_prompt, i, self.n_iterations)
            
            if not plan_extension:
                self.history.append({"iteration": i + 1, "plan": [], "score": last_results['primary_score'], "adopted": False, "action": "noop"})
                continue
            
            print(f"✅ LLM Strategist proposed: {plan_extension}")

            print(f"\nStep 2: Probing the new feature combo plan with {model_name}...")
            # 始终在最好的特征集上进行构建
            df_with_new_features = self.execute_plan(self.best_df.copy(), plan_extension)
            
            new_probe_results = sonata_probe_feature_set(df_with_new_features, self.target_col, self.probe_model_instance, self.model_requires_scaling, self.is_nn)
            new_score = new_probe_results["primary_score"]

            print(f"  - Probe results: {model_name} R2 Score={new_score:.4f}, #Feats: {new_probe_results.get('num_features', -1)}")
            
            print(f"\nStep 3: Deciding whether to adopt the new plan...")
            # 放宽接受阈值，以鼓励更多的探索
            is_adopted = new_score > (self.best_score - 0.01) 
            
            # 奖励是基于与最佳分数的比较
            reward = new_score - self.best_score
            self.experience_buffer.push(current_state_context, plan_extension, reward, is_adopted)
            
            if is_adopted:
                self.last_probe_results = new_probe_results
                print(f"  -> SUCCESS! Plan adopted. New score is {new_score:.4f}.")
                
                # 更新最佳状态
                if new_score > self.best_score:
                    print(f"     -> And it's a NEW PEAK score, beating {self.best_score:.4f}!")
                    self.best_score = new_score
                    self.best_df = df_with_new_features.copy()
                    # 只有当分数提高时才将计划添加到最佳计划中
                    self.best_plan += plan_extension
            else:
                print(f"  -> PLAN REJECTED. Score {new_score:.4f} is not a significant improvement over best score {self.best_score:.4f}. Reverting.")
            
            self.history.append({"iteration": i + 1, "plan": plan_extension, "probe_results": new_probe_results, "adopted": is_adopted, "reward": reward})

        print("\n" + "="*80 + f"\n🏆 T-LAFS ({model_name} Probe) Finished! 🏆")
        print(f"   - Best R² Score Achieved (during search): {self.best_score:.4f}")
        
        # 返回基线分数以便进行迭代贡献分析
        return self.best_df, self.best_plan, self.best_score, self.baseline_score_with_lag1


def main():
    """Main function to run the comprehensive Sonata experiment."""
    # ===== 配置变量 =====
    DATASET_TYPE = 'min_daily_temps'
    N_ITERATIONS = 10
    TARGET_COL = 'temp'

    print("="*80)
    print(f"🚀 Comprehensive Sonata Experiment: Finding Best Features per Model")
    print("="*80)

    # --- 1. 初始化环境和数据 ---
    run_timestamp = datetime.now().strftime("sonata_run_%Y-%m-%d_%H-%M-%S")
    overall_results_dir = os.path.join("results", run_timestamp)
    os.makedirs(overall_results_dir, exist_ok=True)
    print(f"📂 All results for this comprehensive run will be saved in: {overall_results_dir}")

    tlafs_exp.setup_api_client()
    base_df = tlafs_exp.get_time_series_data(DATASET_TYPE)

    # --- 2. 定义我们的模型套件 ---
    # 我们将为这个列表中的每个模型运行一次完整的T-LAFS搜索
    models_to_run = {
        # --- 标准 Scikit-Learn 模型 ---
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0),
        "SVR": SVR(),
        "Lasso": Lasso(random_state=42),
        # --- 神经网络模型 (作为类传入，而不是实例) ---
        "SimpleNN": tlafs_exp.SimpleNN,
        "EnhancedNN": tlafs_exp.EnhancedNN,
        "Transformer": tlafs_exp.TransformerModel,
    }
    
    # 跟踪哪些模型需要对特征进行缩放（不包括NN，因为它们有自己的流程）
    models_requiring_scaling = ['SVR', 'Lasso']
    nn_models = ['SimpleNN', 'EnhancedNN', 'Transformer']

    all_models_summary = []

    # --- 3. 为每个模型运行T-LAFS ---
    for model_name, model_instance in models_to_run.items():
        print("\n" + "#"*100)
        print(f"##  Starting T-LAFS search for: {model_name}")
        print("#"*100 + "\n")

        model_results_dir = os.path.join(overall_results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        # 检查模型类型
        requires_scaling = model_name in models_requiring_scaling
        is_nn = model_name in nn_models

        # 为当前模型初始化T-LAFS算法
        sonata_tlafs = Sonata_TLAFS_Algorithm(
            base_df=base_df.copy(), # 确保每个模型都从原始数据开始
            target_col=TARGET_COL,
            n_iterations=N_ITERATIONS,
            results_dir=model_results_dir,
            probe_model_instance=model_instance,
            model_requires_scaling=requires_scaling,
            is_nn=is_nn
        )
        
        best_df, best_feature_plan, best_score_during_search, baseline_score = sonata_tlafs.run()

        # --- 4. 对找到的最佳特征集进行最终分析 ---
        if best_df is not None:
            print("\n" + "="*40)
            print(f"🔬 FINAL ANALYSIS for {model_name} 🔬")
            print("="*40)
            
            final_probe_results = sonata_probe_feature_set(best_df, TARGET_COL, model_instance, requires_scaling, is_nn)
            final_r2 = final_probe_results['r2']
            
            trained_model = final_probe_results['model']
            X_test = final_probe_results['X_test']
            y_test = final_probe_results['y_test']

            # 计算两种特征重要性
            perm_importance = calculate_permutation_importance(trained_model, X_test, y_test, best_df.drop(columns=['date', TARGET_COL]).columns, model_results_dir, model_name, is_nn)
            iter_importance = calculate_iterative_contribution_importance(sonata_tlafs.history, baseline_score, model_results_dir, model_name)
            
            # 可视化最终预测
            if is_nn:
                # NN的预测值需要逆缩放
                scaler_y = MinMaxScaler().fit(y_test.values.reshape(-1,1)) # 重新拟合一个scaler
                y_pred = trained_model.predict(X_test) # X_test 已经是缩放过的
                y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            else:
                 y_pred_unscaled = trained_model.predict(X_test)

            tlafs_exp.visualize_final_predictions(
                dates=best_df.loc[y_test.index].date, # 从原始DF获取日期
                y_true=y_test,
                y_pred=y_pred_unscaled,
                best_model_name=model_name,
                probe_name=f"{model_name} Probe",
                best_model_metrics={'r2': final_r2, 'mae': 0, 'rmse': 0}, # MAE/RMSE是次要的
                results_dir=model_results_dir
            )

            # 保存该模型的所有结果
            summary_data = {
                'model_name': model_name,
                'search_peak_r2': best_score_during_search,
                'final_r2': final_r2,
                'num_features': len(best_feature_plan),
                'best_feature_plan': best_feature_plan,
                'final_features': list(X_test.columns),
                'permutation_importance': perm_importance.to_dict() if perm_importance is not None else "N/A",
                'iterative_contribution': iter_importance.to_dict() if iter_importance is not None else "N/A",
                'run_history': sonata_tlafs.history
            }
            
            tlafs_exp.save_results_to_json(summary_data, f"tlafs_results_probe_final_{model_name}_results", model_results_dir)
            
            # 添加到最终的总结报告中
            all_models_summary.append({
                'model': model_name,
                'final_r2': final_r2,
                'num_features': len(X_test.columns),
                'search_peak_r2': best_score_during_search
            })

    # --- 5. 打印最终的跨模型总结 ---
    if all_models_summary:
        summary_df = pd.DataFrame(all_models_summary).sort_values('final_r2', ascending=False).set_index('model')
        print("\n" + "="*80)
        print("🏆🏆🏆 Comprehensive Sonata Experiment: Final Summary 🏆🏆🏆")
        print("="*80)
        print(summary_df)
        
        summary_path = os.path.join(overall_results_dir, "sonata_experiment_summary.csv")
        summary_df.to_csv(summary_path)
        print(f"\n✅ Final summary saved to {summary_path}")


if __name__ == "__main__":
    main()
