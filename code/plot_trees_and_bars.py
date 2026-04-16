import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from itertools import combinations
from sklearn.model_selection import train_test_split
import ast
import warnings
warnings.filterwarnings('ignore')

df_fs = pd.read_csv('results/clean_fs.csv')

features = [
    'Gender', 'Age', 'Eye', 'Myopia_Duration', 'Glasses_Duration',
    'IOP', 'Pupil_Light', 'Pupil_Dark', 'Endothelial_Cell', 'Pre_Spherical',
    'Pre_Cylindrical', 'Pre_SE', 'AL', 'LT', 'WTW', 'CCT_Diff', 'Front_Rf',
    'Front_Rs', 'Front_Rm', 'Front_Rper', 'Front_K1', 'Front_K2', 'Front_Km',
    'Front_Astig', 'Front_Rmin', 'Back_Rf', 'Back_Rs', 'Back_Rm', 'Back_Rper',
    'Back_K1', 'Back_K2', 'Back_Km', 'Back_Astig', 'Back_Rmin', 'Pupil_Center',
    'Pachy_Vertex', 'Cornea_Volume', 'Chamber_Volume', 'ACD', 'PD', 'Kmax',
    'Dist_Vertex_Thin', 'Progression_Min', 'Progression_Max', 'Progression_Avg',
    'ART_max', 'Df', 'Db', 'Dp', 'Dt', 'Da', 'D', 'OZ', 'TZ', 'TAZ',
    'Ablation_Depth_Max', 'Ablation_Depth_Center', 'Ablation_Depth_Min', 'RST',
    'Ablation_Volume', 'Pre_BCVA_LogMAR', 'Pre_UCVA_LogMAR'
]

def safe_savefig(fig, filepath):
    try:
        fig.savefig(filepath)
    except PermissionError:
        new_filepath = filepath.replace('.pdf', '_new.pdf')
        print(f"Warning: {filepath} is locked. Saving to {new_filepath} instead.")
        fig.savefig(new_filepath)

def fast_plot(df, dataset_name):
    df_model = df.copy()
    df_model['Gender'] = df_model['Gender'].map({'Male': 1, 'Female': 0})
    df_model['Eye_Right'] = df_model['Eye'].map({'R': 1, 'L': 0, 'OD': 1, 'OS': 0})
    if df_model['Eye_Right'].isna().any():
        df_model['Eye_Right'] = df_model['Eye'].astype('category').cat.codes
    df_model = df_model.dropna(subset=['Post_SE'])

    model_features = [f for f in features if f != 'Eye'] + ['Eye_Right']

    patient_ids = df_model['Patient_ID'].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)

    train_idx = df_model.index[df_model['Patient_ID'].isin(train_ids)]
    X_train = df_model.loc[train_idx, model_features].copy()
    y_train = df_model.loc[train_idx, 'Post_SE']
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)

    res = pd.read_csv(f'results/ML_Importance_{dataset_name}.csv')
    res = res.sort_values(by=['Vote', 'RF_Rank'], ascending=[False, True])
    top_10 = res.head(10)['Feature'].tolist()

    plt.figure(figsize=(10, 6))
    top_10_df = res.head(10).copy()
    ax_bar = sns.barplot(x='Vote', y='Feature', data=top_10_df, color='#4C72B0')
    for p in ax_bar.patches:
        ax_bar.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, xytext=(5, 0), textcoords='offset points')
    plt.xlabel('Ensemble Voting Score (Max 3)')
    plt.title(f'Top 10 Feature Importance - {dataset_name}')
    plt.xlim(0, 3.5)
    plt.xticks([0, 1, 2, 3])
    plt.tight_layout()
    safe_savefig(plt.gcf(), f'figures/Feature_Importance_Bar_{dataset_name}.pdf')
    plt.close()

    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X_train[top_10], y_train)

    fig_dt, ax_dt = plt.subplots(figsize=(20, 12))
    annotations_dt = plot_tree(dt, feature_names=top_10, filled=True, rounded=True, fontsize=9, ax=ax_dt)

    for a in annotations_dt:
        if a.get_bbox_patch() is None:
            a.remove()

    node_ann = [a for a in annotations_dt if a.get_bbox_patch() is not None]

    tree_dt = dt.tree_
    for i in range(tree_dt.node_count):
        if tree_dt.children_left[i] != tree_dt.children_right[i]:
            left_child = tree_dt.children_left[i]
            right_child = tree_dt.children_right[i]
            px, py = node_ann[i].get_position()
            lx, ly = node_ann[left_child].get_position()
            rx, ry = node_ann[right_child].get_position()
            ax_dt.text((px + lx)/2, (py + ly)/2, 'True', ha='center', va='center', fontsize=10, color='darkgreen', weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
            ax_dt.text((px + rx)/2, (py + ry)/2, 'False', ha='center', va='center', fontsize=10, color='darkred', weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
    plt.title(f'Decision Tree - {dataset_name}')
    plt.tight_layout()
    safe_savefig(plt.gcf(), f'figures/DecisionTree_{dataset_name}.pdf')
    plt.close(fig_dt)

    df_model['Outcome_Class'] = np.where(df_model['Post_SE'] > -0.5, 1, 0)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train[top_10], df_model.loc[train_idx, 'Outcome_Class'])

    fig_clf, ax_clf = plt.subplots(figsize=(20, 12))
    annotations_clf = plot_tree(clf, feature_names=top_10, class_names=['Poor', 'Good'], filled=True, rounded=True, fontsize=9, ax=ax_clf)

    for a in annotations_clf:
        if a.get_bbox_patch() is None:
            a.remove()

    node_ann_clf = [a for a in annotations_clf if a.get_bbox_patch() is not None]

    tree_clf = clf.tree_
    for i in range(tree_clf.node_count):
        if tree_clf.children_left[i] != tree_clf.children_right[i]:
            left_child = tree_clf.children_left[i]
            right_child = tree_clf.children_right[i]
            px, py = node_ann_clf[i].get_position()
            lx, ly = node_ann_clf[left_child].get_position()
            rx, ry = node_ann_clf[right_child].get_position()
            ax_clf.text((px + lx)/2, (py + ly)/2, 'True', ha='center', va='center', fontsize=10, color='darkgreen', weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
            ax_clf.text((px + rx)/2, (py + ry)/2, 'False', ha='center', va='center', fontsize=10, color='darkred', weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
    plt.title(f'Classification Tree - {dataset_name}')
    plt.tight_layout()
    safe_savefig(plt.gcf(), f'figures/ClassificationTree_{dataset_name}.pdf')
    plt.close(fig_clf)

    rf_params_str = res['RF_Best_Params'].iloc[0]
    rf_params = ast.literal_eval(rf_params_str)
    rf_params['n_estimators'] = 100
    rf = RandomForestRegressor(random_state=42, **rf_params)
    rf.fit(X_train, y_train)

    top_5 = top_10[:5]
    top_5_idx = [X_train.columns.get_loc(f) for f in top_5]
    pair_features = list(combinations(top_5_idx, 2))
    fig, ax = plt.subplots(4, 3, figsize=(18, 20))
    PartialDependenceDisplay.from_estimator(rf, X_train, pair_features, ax=ax.flatten()[:len(pair_features)])
    plt.suptitle(f'Pairwise Partial Dependence Plots - {dataset_name}')
    plt.tight_layout()
    safe_savefig(plt.gcf(), f'figures/PDP_{dataset_name}.pdf')
    plt.close()

    print(f"Fast plots generated for {dataset_name}")

if __name__ == '__main__':
    fast_plot(df_fs, 'FSLASIK')
