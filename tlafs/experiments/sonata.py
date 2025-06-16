"""
Sonata实验模块 - 使用重构后的TLAFS框架
"""
import os
from datetime import datetime
import json

# 从重构后的tlafs框架中导入必要的模块
from tlafs.core.algorithm import TLAFS_Algorithm
from tlafs.utils.data_utils import get_time_series_data
from tlafs.features.generator import execute_plan
from tlafs.utils.evaluation import probe_feature_set, evaluate_on_multiple_models
from tlafs.visualization.plotting import visualize_final_predictions
from tlafs.utils.file_utils import save_results # 假设这个函数仍然适用
from tlafs.analysis.feature_importance import calculate_permutation_importance, analyze_iterative_contribution
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import r2_score
import pandas as pd

def main():
    """主函数，运行完整的Sonata T-LAFS实验。"""
    
    # ===== 配置变量 =====
    DATASET_TYPE = 'min_daily_temps'
    N_ITERATIONS = 7
    TARGET_COL = 'temp'
    
    print("="*80)
    print(f"🚀 T-LAFS Sonata实验: 使用模块化框架")
    print("="*80)

    # --- 1. 初始化环境和数据 ---
    run_timestamp = datetime.now().strftime("sonata_%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", run_timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"📂 本次运行的所有结果将保存在: {results_dir}")

    base_df = get_time_series_data(DATASET_TYPE)

    # --- 2. 运行 T-LAFS 特征搜索 ---
    tlafs_alg = TLAFS_Algorithm(
        base_df=base_df,
        target_col=TARGET_COL,
        n_iterations=N_ITERATIONS,
        results_dir=results_dir
    )
    
    # TLAFS_Algorithm的__init__现在会自动调用预训练
    
    # 定义TLAFS运行所需的参数字典
    # 这些参数会被传递给 execute_plan 函数
    tlafs_params = {
        "target_col_static": tlafs_alg.target_col_static,
        "pretrain_cols_static": tlafs_alg.pretrain_cols_static,
        "pretrained_encoders": tlafs_alg.pretrained_encoders,
        "embedder_scalers": tlafs_alg.embedder_scalers,
        # 如果有其他模型，也在这里添加
        # "meta_forecast_models": tlafs_alg.meta_forecast_models,
        # "meta_scalers": tlafs_alg.meta_scalers,
    }

    # 将执行和探测函数传递给run方法
    best_df, best_feature_plan, best_score_during_search = tlafs_alg.run(
        execute_plan_func=lambda df, plan, params: execute_plan(df, plan, params),
        probe_func=lambda df, target, features: probe_feature_set(df, target, features)
    )

    # --- 3. 对找到的最佳特征集进行最终分析 ---
    if best_df is not None:
        probe_name_for_reporting = "Sonata_Transformer_Specialist"
        print("\n" + "="*40)
        print(f"🔬 对所有模型进行最终验证 ({probe_name_for_reporting}) 🔬")
        print("="*40)
        
        final_metrics, final_results = evaluate_on_multiple_models(
            best_df,
            TARGET_COL
        )

        if final_metrics:
            best_final_model_name = max(final_metrics, key=lambda k: final_metrics[k]['r2'])
            best_final_metrics = final_metrics[best_final_model_name]
            
            # 从final_results中为最佳模型获取日期、真实值和预测值
            # 注意：evaluate_on_multiple_models返回的y_test是列表，需要转换为pd.Series以便使用索引
            best_result_data = final_results[best_final_model_name]
            
            # 我们需要从best_df中获取正确的日期索引
            # evaluate_on_multiple_models内部进行了train_test_split，我们需要获取测试集的索引
            # 为了简化，我们假设evaluate_on_multiple_models返回的y_true的索引与best_df的测试部分匹配
            # 一个更健壮的方法是让评估函数也返回测试索引
            test_indices = best_df.index[-len(best_result_data['y_true']):]


            visualize_final_predictions(
                dates=best_df.loc[test_indices, 'date'],
                y_true=best_result_data['y_true'],
                y_pred=best_result_data['y_pred'],
                best_model_name=best_final_model_name,
                probe_name=probe_name_for_reporting,
                best_model_metrics=best_final_metrics,
                results_dir=results_dir
            )

            # --- 3.5. 特征重要性分析 ---
            print("\n" + "="*40)
            print(f"🔬 特征重要性分析 🔬")
            print("="*40)

            # a) 迭代边际贡献分析
            print("\n--- 迭代边际贡献 (Iterative Marginal Contribution) ---")
            print("此分析显示了在T-LAFS搜索过程中，按顺序添加每个特征时模型性能的变化。")
            iterative_contribution_df = analyze_iterative_contribution(tlafs_alg.history)
            if not iterative_contribution_df.empty:
                print(iterative_contribution_df.to_string())
                # 保存到CSV
                iterative_contribution_path = os.path.join(results_dir, "iterative_contribution.csv")
                iterative_contribution_df.to_csv(iterative_contribution_path, index=False)
                print(f"\n✅ 迭代贡献分析已保存至: {iterative_contribution_path}")

            # b) 排列重要性分析
            print("\n--- 排列重要性 (Permutation Importance) ---")
            print("此分析显示了在最终模型中，随机打乱每个特征后对模型性能（R^2）造成的负面影响。")
            
            # 为了进行此分析，我们需要一个训练好的模型和验证集
            # 我们将使用LightGBM，因为它速度快且性能好
            features = [col for col in best_df.columns if col not in ['date', TARGET_COL]]
            X = best_df[features]
            y = best_df[TARGET_COL]

            # 简单拆分数据用于分析
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # 训练一个LGBM模型
            lgbm_for_importance = lgb.LGBMRegressor(random_state=42)
            lgbm_for_importance.fit(X_train, y_train)

            # 计算排列重要性
            permutation_importance_df = calculate_permutation_importance(
                model=lgbm_for_importance,
                X_val=X_val,
                y_val=y_val,
                metric_func=r2_score
            )
            print(permutation_importance_df.to_string())

            # 保存到CSV
            perm_importance_path = os.path.join(results_dir, "permutation_importance.csv")
            permutation_importance_df.to_csv(perm_importance_path)
            print(f"\n✅ 排列重要性分析已保存至: {perm_importance_path}")

            # --- 3.6. 基于重要性的最终特征剪枝 ---
            print("\n========================================\n🔪 基于重要性的最终特征剪枝 🔪\n========================================")
            PRUNING_THRESHOLD = 0.0
            print(f"剪枝阈值 (Permutation Importance): <= {PRUNING_THRESHOLD}")
            
            # 获取低重要性特征
            features_to_drop = permutation_importance_df[
                permutation_importance_df['importance_mean'] <= PRUNING_THRESHOLD
            ].index.tolist()
            
            if features_to_drop:
                print(f"\n将删除以下低重要性特征:")
                for feature in features_to_drop:
                    print(f"  - {feature}")
                
                # 删除低重要性特征
                df_pruned = best_df.drop(columns=features_to_drop)
                
                # 重新评估剪枝后的模型性能
                print("\n重新评估剪枝后的模型性能...")
                final_metrics_pruned, final_results_pruned = evaluate_on_multiple_models(
                    df_pruned,
                    TARGET_COL
                )

                if final_metrics_pruned:
                    best_model_name_pruned = max(final_metrics_pruned, key=lambda k: final_metrics_pruned[k]['r2'])
                    best_metrics_pruned = final_metrics_pruned[best_model_name_pruned]
                    r2_pruned = best_metrics_pruned['r2']
                    
                    print(f"\n剪枝后的性能 (最佳模型: {best_model_name_pruned}):")
                    print(f"  - 特征数量: {len(df_pruned.columns) - 2} (从 {len(best_df.columns) - 2} 减少了 {len(features_to_drop)} 个)")
                    print(f"  - 新的 R² 分数: {r2_pruned:.4f} (原始最佳 R²: {best_final_metrics['r2']:.4f})")

                    # 如果性能没有显著下降，使用剪枝后的特征集
                    if r2_pruned >= best_final_metrics['r2'] - 0.01:  # 允许最多0.01的性能下降
                        print("\n✅ 采用剪枝后的特征集。将更新最终结果。")
                        best_df = df_pruned
                        final_metrics = final_metrics_pruned
                        final_results = final_results_pruned
                        # 更新用于报告和可视化的最佳模型变量
                        best_final_model_name = best_model_name_pruned
                        best_final_metrics = best_metrics_pruned
                        best_result_data = final_results[best_final_model_name]
                    else:
                        print(f"\n❌ 剪枝后性能下降超过阈值。将保持原始特征集。")
                else:
                    print("\n⚠️ 剪枝后模型评估失败。将保持原始特征集。")
            else:
                print("\n没有发现可以删除的低重要性或无用特征。")

            # --- 4. 保存所有结果 ---
            print("\n" + "="*40)
            print(f"💾 保存最终结果与可视化 💾")
            print("="*40)
            
            # 现在, 所有变量 (best_df, best_final_metrics, etc.) 都反映了剪枝决策后的最终状态
            
            # 重新获取测试集索引，以防剪枝导致df变化
            test_indices_final = best_df.index[-len(best_result_data['y_true']):]

            # 使用最终确定的数据进行可视化
            visualize_final_predictions(
                dates=best_df.loc[test_indices_final, 'date'],
                y_true=best_result_data['y_true'],
                y_pred=best_result_data['y_pred'],
                best_model_name=best_final_model_name,
                probe_name=f"{probe_name_for_reporting}_Final", # 添加后缀以区分
                best_model_metrics=best_final_metrics,
                results_dir=results_dir,
                filename_suffix="_post_pruning" # 添加文件名后缀
            )

            summary_data = {
                "probe_model": probe_name_for_reporting,
                "best_score_during_search": best_score_during_search,
                "best_feature_plan": best_feature_plan,
                "final_features": [col for col in best_df.columns if col not in ['date', TARGET_COL]],
                "final_validation_scores_all_models": final_metrics,
                "best_final_model": {
                    "name": best_final_model_name,
                    "metrics": best_final_metrics
                },
                "run_history": tlafs_alg.history
            }
            
            # 使用一个通用的JSON保存函数
            results_path = os.path.join(results_dir, "sonata_tlafs_summary.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=4, default=str) # 使用default=str处理Numpy类型
            print(f"✅ 最终总结已保存至: {results_path}")

    else:
        print("\nT-LAFS未能找到有效的特征计划。")

if __name__ == "__main__":
    main()