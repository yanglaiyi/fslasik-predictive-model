import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.utils import resample
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

def run_ml_pipeline(df, dataset_name):
    print(f"Starting pipeline for {dataset_name}")
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
    test_idx = df_model.index[df_model['Patient_ID'].isin(test_ids)]

    X_train = df_model.loc[train_idx, model_features].copy()
    X_test = df_model.loc[test_idx, model_features].copy()
    y_train = df_model.loc[train_idx, 'Post_SE']
    y_test = df_model.loc[test_idx, 'Post_SE']

    train_medians = X_train.median(numeric_only=True)
    missing_cols = X_train.columns[X_train.isna().any() | X_test.isna().any()]
    if len(missing_cols) > 0:
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

    cv5x10 = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    lasso_counts = np.zeros(X_train.shape[1])
    for i in range(100):
        X_b, y_b = resample(X_train, y_train, random_state=i)
        lasso = LassoCV(cv=5, random_state=i).fit(X_b, y_b)
        lasso_counts += (lasso.coef_ != 0)

    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={
            'n_estimators': [200, 400],
            'max_depth': [4, 6, None],
            'min_samples_leaf': [1, 3]
        },
        cv=cv5x10,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    rf_grid.fit(X_train, y_train)
    rf = rf_grid.best_estimator_
    rf_imp = rf.feature_importances_

    xgb_grid = GridSearchCV(
        XGBRegressor(random_state=42, objective='reg:squarederror'),
        param_grid={
            'n_estimators': [200, 400],
            'max_depth': [3, 5],
            'learning_rate': [0.03, 0.1]
        },
        cv=cv5x10,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    xgb_grid.fit(X_train, y_train)
    xgb = xgb_grid.best_estimator_
    xgb_imp = xgb.feature_importances_

    res = pd.DataFrame({
        'Feature': model_features,
        'LASSO_Freq': lasso_counts,
        'RF_Imp': rf_imp,
        'XGB_Imp': xgb_imp
    })

    res['RF_Rank'] = res['RF_Imp'].rank(ascending=False)
    res['XGB_Rank'] = res['XGB_Imp'].rank(ascending=False)
    res['LASSO_Rank'] = res['LASSO_Freq'].rank(ascending=False, method='min')

    res['Vote'] = (res['RF_Rank'] <= 15).astype(int) + (res['XGB_Rank'] <= 15).astype(int) + (res['LASSO_Rank'] <= 15).astype(int)

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
    plt.savefig(f'figures/Feature_Importance_Bar_{dataset_name}.pdf')
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
    plt.savefig(f'figures/DecisionTree_{dataset_name}.pdf')
    plt.close(fig_dt)

    top_5 = top_10[:5]
    top_5_idx = [X_train.columns.get_loc(f) for f in top_5]
    pair_features = list(combinations(top_5_idx, 2))
    fig, ax = plt.subplots(4, 3, figsize=(18, 20))
    PartialDependenceDisplay.from_estimator(rf, X_train, pair_features, ax=ax.flatten()[:len(pair_features)])
    plt.suptitle(f'Pairwise Partial Dependence Plots - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'figures/PDP_{dataset_name}.pdf')
    plt.close()

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
    plt.savefig(f'figures/ClassificationTree_{dataset_name}.pdf')
    plt.close(fig_clf)

    reg_tree_features = {top_10[i] for i in dt.tree_.feature if i >= 0}
    clf_tree_features = {top_10[i] for i in clf.tree_.feature if i >= 0}
    res['In_Regression_Tree_Path'] = res['Feature'].isin(reg_tree_features).astype(int)
    res['In_Classification_Tree_Path'] = res['Feature'].isin(clf_tree_features).astype(int)
    res['RF_Best_Params'] = str(rf_grid.best_params_)
    res['XGB_Best_Params'] = str(xgb_grid.best_params_)
    res.to_csv(f'results/ML_Importance_{dataset_name}.csv', index=False)
    print(f"ML Importance saved for {dataset_name}")

    corr = X_train[top_10].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix of Top 10 Features - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'figures/CorrMatrix_{dataset_name}.pdf')
    plt.close()

    return top_10

if __name__ == '__main__':
    print("Starting data_analysis_part4.py execution...")
    try:
        fs_top = run_ml_pipeline(df_fs, 'FSLASIK')

        with open('results/top_features.txt', 'w') as f:
            f.write(f"FSLASIK: {fs_top}\n")
        print("ML Pipeline completed for FSLASIK.")
    except Exception as e:
        print(f"Error: {e}")
