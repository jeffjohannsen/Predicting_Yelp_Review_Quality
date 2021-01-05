
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sqlalchemy import create_engine
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from model_pipeline import ModelPipeline



def print_data_info(target, features):
    """
    Basic info about data class balance.

    Args:
        target (Series): Pandas Series of binary int 0,1 values.
        features (Dataframe):
    """
    print(f'Target Shape: {target.shape}')
    quality_reviews = sum(target)
    not_quality_reviews = target.shape[0] - quality_reviews
    print(f'Quality: {quality_reviews}     Not Quality: {not_quality_reviews}')
    percent_quality = (quality_reviews / target.shape[0])*100
    print(f'Percent of reviews that are quality: {percent_quality:.0f}%')
    print(target.head(10))
    print(features.info())


def autolabel(rects, axe, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    xpos indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        axe.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                 '{}%'.format(height), ha=ha[xpos], va='bottom', fontsize=20,
                 weight='bold')


def create_model_performance_metrics(forest, X_train, X_test, y_train, y_test):
    """
    Prints and returns common model performance metrics.

    Args:
        forest (object): Fitted Random Forest Classifier
        X_train (Dataframe): Features
        X_test (Dataframe): Features
        y_train (Series): Targets
        y_test (Series): Targets

    Returns:
        Tuple of Lists: 4 Lists - model_results,
                                - model_results_labels
                                - performance metrics
                                - performance metric labels
    """
    # oobscore = forest.oob_score_
    # print(f'Out-of-Bag Score: {oobscore:.2f}')

    train_accuracy_score = forest.score(X_train, y_train)
    print(f'Train Accuracy: {train_accuracy_score:.2f}')

    test_accuracy_score = forest.score(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy_score:.2f}')

    y_pred = forest.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred)

    total = sum([tn, fp, fn, tp])
    model_results = [tp/total, tn/total, fp/total, fn/total]
    rounded_model_results = [round(num*100) for num in model_results]
    model_results_labels = ['True Positives', 'True Negatives',
                            'False Positives', 'False Negatives']
    print(rounded_model_results)

    performance_metrics = [accuracy, precision, recall, F1_score]
    rounded_performance_metrics = [round(num*100) for num
                                   in performance_metrics]
    performance_metrics_labels = ['Accuracy', 'Precision',
                                  'Recall', 'F1 Score']
    print(rounded_performance_metrics)
    return (rounded_model_results, model_results_labels,
            rounded_performance_metrics, performance_metrics_labels)


def bar_color_chooser(x):
    """
    Helper function for bar graph bar colors.
    """
    if x.startswith('User'):
        return 'tab:red'
    elif x.startswith('Business'):
        return 'tab:gray'
    elif x.startswith('Review Text'):
        return 'tab:orange'
    elif x.startswith('Review'):
        return 'tab:blue'
    else:
        return 'tab:purple'


def load_data_from_yelp_2(table, num_records):
    """
    Connects to yelp_2 database on Postgres and
    loads data from specified tables and record count.

    Args:
        table (str): Name of table in yelp_2.
        num_records (int): Number of records to load.

    Returns:
        Dataframe: Pandas dataframe of records
                from yelp_2 database.
    """
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp_2'
    engine = create_engine(connect)
    query = f'''
            SELECT *
            FROM {table}
            LIMIT {num_records}
            ;
            '''
    df = pd.read_sql(query, con=engine)
    df = df.copy()
    return df


def plot_model_performance(results, metrics, save=False):
    """
    Creates plots showing the results of testing
    a binary classification model.

    Args:
        results (list-like of numeric): Order Matters.
            True Positives, True Negatives,
            False Positives, False Negatives

        metrics (list-like of numeric): Order Matters.
            Accuracy, Precision, Recall, F1 Score

        save (bool, optional): Whether or not to save the plots to file.
                                Defaults to False.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    # Data
    x_data_ax1 = np.arange(4)
    x_data_ax2 = np.arange(4)
    y_data_ax1 = results
    y_data_ax2 = metrics
    # Labels
    model_results_labels = ['True Positives', 'True Negatives',
                            'False Positives', 'False Negatives']
    performance_metrics_labels = ['Accuracy', 'Precision',
                                  'Recall', 'F1 Score']
    # Plotting model results
    bar_colors_ax1 = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red']
    rects1 = ax1.bar(x_data_ax1, y_data_ax1, color=bar_colors_ax1)
    ax1.set_title("Model Results", fontweight="bold")
    ax1.set_ylabel("Percent of Model Predictions", fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x_data_ax1)
    ax1.set_xticklabels(model_results_labels, rotation=45,
                        ha="right", fontweight='bold')
    autolabel(rects1, ax1, "center")
    # Plotting performance metrics
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
    if save:
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_permutation_importances(forest, X_train, y_train, num=20, save=False):
    """
    Plots permutation importances for a fitted random forest.

    Args:
        forest (object): Fitted random forest model.
        X_train (Dataframe): Training features.
        y_train (Series): Training target.
        save (bool, optional): Whether to save plot to file.
                               Defaults to False.
    """
    result = permutation_importance(forest, X_train, y_train, n_repeats=5,
                                    random_state=None)
    perm_sorted_idx = result.importances_mean.argsort()[::-1]
    pi_labels = [feature.replace('_', ' ').title() for
                 feature in X_train.columns[perm_sorted_idx]][:num]
    x_data_pi = np.arange(len(X_train.columns))[:num]
    y_data_pi = result.importances_mean[perm_sorted_idx].T[:num]
    fig, ax = plt.subplots(figsize=(12, 9))
    bar_colors_pi = list(map(bar_color_chooser, pi_labels))
    ax.bar(x_data_pi, y_data_pi, color=bar_colors_pi)
    ax.set_title("Permutation Importances", fontweight="bold")
    ax.set_ylabel("Importance Score", fontweight="bold")
    ax.set_xticks(x_data_pi)
    ax.set_xticklabels(pi_labels, rotation=45, ha="right",
                       fontweight='semibold')
    legend_elements = [Line2D([0], [0], color='tab:red', lw=20,
                              label='User Data'),
                       Line2D([0], [0], color='tab:gray', lw=20,
                              label='Business Data'),
                       Line2D([0], [0], color='tab:blue', lw=20,
                              label='Review MetaData'),
                       Line2D([0], [0], color='tab:orange', lw=20,
                              label='Review Text Data')]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    plt.show()
    if save:
        plt.savefig('permutation_importances.png', dpi=300,
                    bbox_inches='tight')
    else:
        plt.show()


def plot_feature_importances(forest, X_train, num=20, save=False):
    """
    Plots feature importances for a fitted random forest.

    Args:
        forest (object): Fitted random forest model.
        X_train (Dataframe): Training features.
        save (bool, optional): Whether to save plot to file.
                               Defaults to False.
    """
    tree_importance_sorted_idx = np.argsort(forest.feature_importances_)[::-1]
    x_data_fi = np.arange(len(X_train.columns))[:num]
    y_data_fi = forest.feature_importances_[tree_importance_sorted_idx][:num]
    fi_labels = [feature.replace('_', ' ').title() for
                 feature in X_train.columns[tree_importance_sorted_idx]][:num]
    fig, ax = plt.subplots(figsize=(12, 9))
    bar_colors_fi = list(map(bar_color_chooser, fi_labels))
    ax.bar(x_data_fi, y_data_fi, color=bar_colors_fi)
    ax.set_title("Feature Importances", fontweight="bold")
    ax.set_ylabel("Importance Score", fontweight="bold")
    ax.set_xticks(x_data_fi)
    ax.set_xticklabels(fi_labels, rotation=45, ha="right",
                       fontweight='semibold')
    legend_elements = [Line2D([0], [0], color='tab:red', lw=20,
                              label='User Data'),
                       Line2D([0], [0], color='tab:gray', lw=20,
                              label='Business Data'),
                       Line2D([0], [0], color='tab:blue', lw=20,
                              label='Review MetaData'),
                       Line2D([0], [0], color='tab:orange', lw=20,
                              label='Review Text Data')]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    if save:
        plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_feature_correlation(X_train):
    """
    Plots and prints correlation data between features.

    Args:
        X_train (Dataframe): Features data.
    """
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


if __name__ == "__main__":
    # Pandas and Matplotlib global options
    pd.set_option('display.max_columns', 100)
    pd.set_option("max_rows", 1000)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 16, 'font.family': 'sans'})

    pipeline = ModelPipeline(False)
    X_train, y_train, X_test, y_test, fitted_forest = \
        pipeline.run_full_pipeline(use_cv=False, print_results=True,
                                   save_results=True, question='td',
                                   records=100, data='both',
                                   target='T2_CLS_ufc_>0', model='Forest Cls',
                                   scalar='no_scaling', balancer=None)
    print('Training Data:')
    print_data_info(y_train, X_train)
    print('Testing Data:')
    print_data_info(y_test, X_test)
    print()
    print('Testing Results:')
    (results, results_labels, metrics, metrics_labels) = \
        create_model_performance_metrics(fitted_forest, X_train, X_test,
                                         y_train, y_test)
    plot_model_performance(results, metrics)
    plot_feature_importances(fitted_forest, X_train)
    plot_permutation_importances(fitted_forest, X_train, y_train)
    plot_feature_correlation(X_train)
