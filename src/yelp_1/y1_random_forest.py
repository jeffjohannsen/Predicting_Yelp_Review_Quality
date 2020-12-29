
# ! WARNING: Work in Progress. Testing and Documentation missing.
# TODO: Combine with model pipeline from model_prep.py.
# TODO: Combine code blocks into functions and classes.

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
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


def connect_to_cls_train_data(num_records):
    """
    Connect to yelp database > model_data_cls_train table
    and grab records.

    Args:
        num_records (int): Number of records to return.
                            Maps to a SQL LIMIT command.

    Returns:
        Dataframe: Records Requested
    """
    connect = 'postgresql+psycopg2://postgres:password@localhost:5432/yelp'
    engine = create_engine(connect)
    query = f'''
            SELECT *
            FROM model_data_cls_train
            LIMIT {num_records}
            ;
            '''
    df = pd.read_sql(query, con=engine)
    data = df.copy()
    data = data.drop_duplicates(subset='review_id')
    return data


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


def model_performace_metrics(forest, X_train, X_test, y_train, y_test):
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
    oobscore = forest.oob_score_
    print(f'Out-of-Bag Score: {oobscore:.2f}')

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
    first_three = x[:3]
    if first_three == 'Use':
        return 'tab:blue'
    elif first_three == 'Rev':
        return 'tab:gray'
    else:
        return 'tab:red'


if __name__ == "__main__":
    # Pandas and Matplotlib global options
    pd.set_option('display.max_columns', 100)
    pd.set_option("max_rows", 1000)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)

    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 16, 'font.family': 'sans'})

    data = connect_to_cls_train_data(100000)

    target = data['TARGET_review_has_upvotes']
    unused_features = ['level_0', 'index', 'review_id',
                       'restaurant_latitude',
                       'restaurant_longitude',
                       'TARGET_review_has_upvotes']
    # Non-correlation corrected. All features.
    features = data.drop(labels=unused_features, axis=1)

    # Correlation corrected features. Subset of All features.
    # features = ['review_stars', 'restaurant_overall_stars',
    #             'restaurant_review_count',
    #             'restaurant_is_open', 'restaurant_price',
    #             'user_average_stars_given',
    #             'user_review_count', 'user_friend_count',
    #             'user_years_since_last_elite',
    #             'user_days_active_at_review_time']

    print_data_info(target=target, features=features)

    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.20,
                                                        random_state=None)

    forest = RandomForestClassifier(n_estimators=100,
                                    criterion='gini',
                                    max_depth=None,
                                    max_features='sqrt',
                                    max_leaf_nodes=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    min_impurity_decrease=0.0,
                                    max_samples=None,
                                    random_state=None,
                                    oob_score=True,
                                    verbose=2
                                    )

    forest.fit(X_train, y_train)

    rounded_model_results = model_performace_metrics(forest, X_train,
                                                     X_test, y_train,
                                                     y_test)[0]
    model_results_labels = model_performace_metrics(forest, X_train,
                                                    X_test, y_train,
                                                    y_test)[1]
    rounded_performance_metrics = model_performace_metrics(forest, X_train,
                                                           X_test, y_train,
                                                           y_test)[2]
    performance_metrics_labels = model_performace_metrics(forest, X_train,
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
    x_data_fi = np.arange(len(features.columns))
    x_data_pi = np.arange(len(features.columns))
    y_data_fi = forest.feature_importances_[tree_importance_sorted_idx]
    y_data_pi = result.importances_mean[perm_sorted_idx].T
    fi_labels = [feature.replace('_', ' ').title() for
                 feature in features.columns[tree_importance_sorted_idx]]
    pi_labels = [feature.replace('_', ' ').title() for
                 feature in features.columns[perm_sorted_idx]]

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
                              label='Restaurant Data'),
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
    corr = spearmanr(features).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage,
                                  labels=features.columns.tolist(),
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
    print(features.columns[selected_features])
