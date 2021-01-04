
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D

from model_pipeline import ModelDetailsStorage, ModelPipeline

from random_forest import (model_performance_metrics, autolabel, bar_color_chooser)

if __name__ == "__main__":
    for data in ['text', 'non_text', 'both']:
        for target in ['T2_CLS_ufc_>0']:
            for model in ['Forest Cls']:
                pipeline = ModelPipeline(run_on_aws=False)
                X_train, y_train, X_test, y_test, Model = \
                    pipeline.run_full_pipeline(use_cv=False,
                                               print_results=True,
                                               save_results=True,
                                               question='td',
                                               records=10000,
                                               data=data,
                                               target=target,
                                               model=model,
                                               scalar='power',
                                               balancer='smote')
                
                forest = Model

                rounded_model_results = model_performance_metrics(forest, X_train,
                                                     X_test, y_train,
                                                     y_test)[0]
                model_results_labels = model_performance_metrics(forest, X_train,
                                                                X_test, y_train,
                                                                y_test)[1]
                rounded_performance_metrics = model_performance_metrics(forest, X_train,
                                                                    X_test, y_train,
                                                                    y_test)[2]
                performance_metrics_labels = model_performance_metrics(forest, X_train,
                                                                    X_test, y_train,
                                                                    y_test)[3]

                def create_model_performance_plot(save=False):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

                    # data
                    x_data_ax1 = np.arange(4)
                    x_data_ax2 = np.arange(4)
                    y_data_ax1 = rounded_model_results
                    y_data_ax2 = rounded_performance_metrics

                    # plotting model results
                    bar_colors_ax1 = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red']
                    rects1 = ax1.bar(x_data_ax1, y_data_ax1, color=bar_colors_ax1)
                    ax1.set_title("Model Results", fontweight="bold")
                    ax1.set_ylabel("Percent of Model Predictions", fontweight="bold")
                    ax1.set_ylim(0, 105)
                    ax1.set_xticks(x_data_ax1)
                    ax1.set_xticklabels(model_results_labels, rotation=45,
                                        ha="right", fontweight='bold')
                    autolabel(rects1, ax1, "center")

                    # plotting performance metrics
                    bar_colors_ax2 = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue']
                    rects2 = ax2.bar(x_data_ax2, y_data_ax2, color=bar_colors_ax2)
                    ax2.set_title("Model Performance Metrics", fontweight="bold")
                    ax2.set_ylabel("Metric Score", fontweight="bold")
                    ax2.set_ylim(0, 105)
                    ax2.set_xticks(x_data_ax2)
                    ax2.set_xticklabels(performance_metrics_labels, rotation=45,
                                        ha="right", fontweight='bold')
                    autolabel(rects2, ax2, "center")

                    fig.tight_layout()
                    plt.show()
                    if save:
                        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')

                # Permutation Importances
                result = permutation_importance(forest, X_train, y_train, n_repeats=5,
                                                random_state=None)
                perm_sorted_idx = result.importances_mean.argsort()[::-1]

                tree_importance_sorted_idx = np.argsort(forest.feature_importances_)[::-1]
                tree_indices = np.arange(0, len(forest.feature_importances_)) + 0.5

                # graph data
                x_data_fi = np.arange(len(X_train.columns))
                x_data_pi = np.arange(len(X_train.columns))
                y_data_fi = forest.feature_importances_[tree_importance_sorted_idx]
                y_data_pi = result.importances_mean[perm_sorted_idx].T
                fi_labels = [feature.replace('_', ' ').title() for
                            feature in X_train.columns[tree_importance_sorted_idx]]
                pi_labels = [feature.replace('_', ' ').title() for
                            feature in X_train.columns[perm_sorted_idx]]

                # plotting feature importances
                fig, ax = plt.subplots(figsize=(12, 9))
                bar_colors_fi = list(map(bar_color_chooser, fi_labels))
                ax.bar(x_data_fi, y_data_fi, color=bar_colors_fi)
                ax.set_title("Feature Importances", fontweight="bold")
                ax.set_ylabel("Importance Score", fontweight="bold")
                ax.set_xticks(x_data_fi)
                ax.set_xticklabels(fi_labels, rotation=45, ha="right",
                                fontweight='semibold')
                legend_elements = [Line2D([0], [0], color='tab:blue', lw=20,
                                        label='User Data'),
                                Line2D([0], [0], color='tab:red', lw=20,
                                        label='Business Data'),
                                Line2D([0], [0], color='tab:gray', lw=20,
                                        label='Review Data')]

                ax.legend(handles=legend_elements)

                fig.tight_layout()
                plt.show()
                # plt.savefig('feature_importances.png', dpi = 300, bbox_inches='tight')

                # plotting permutation importances
                fig, ax = plt.subplots(figsize=(12, 9))
                bar_colors_pi = list(map(bar_color_chooser, pi_labels))
                ax.bar(x_data_pi, y_data_pi, color=bar_colors_pi)
                ax.set_title("Permutation Importances", fontweight="bold")
                ax.set_ylabel("Importance Score", fontweight="bold")
                ax.set_xticks(x_data_pi)
                ax.set_xticklabels(pi_labels, rotation=45, ha="right",
                                fontweight='semibold')
                legend_elements = [Line2D([0], [0], color='tab:blue', lw=20,
                                label='User Data'),
                                Line2D([0], [0], color='tab:red', lw=20,
                                label='Restaurant Data'),
                                Line2D([0], [0], color='tab:gray', lw=20,
                                label='Review Data')]

                ax.legend(handles=legend_elements)

                fig.tight_layout()
                plt.show()
                # plt.savefig('permutation_importances.png', dpi = 300,
                #              bbox_inches='tight')

                # Feature Correlation
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
                corr = spearmanr(X_train).correlation
                corr_linkage = hierarchy.ward(corr)
                dendro = hierarchy.dendrogram(corr_linkage,
                                            labels=X_train.columns.tolist(),
                                            ax=ax1, leaf_rotation=90)
                dendro_idx = np.arange(0, len(dendro['ivl']))

                ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
                ax2.set_xticks(dendro_idx)
                ax2.set_yticks(dendro_idx)
                ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
                ax2.set_yticklabels(dendro['ivl'])
                fig.tight_layout()
                plt.show()

                cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
                cluster_id_to_feature_ids = defaultdict(list)
                for idx, cluster_id in enumerate(cluster_ids):
                    cluster_id_to_feature_ids[cluster_id].append(idx)
                selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
                print(X_train.columns[selected_features])
