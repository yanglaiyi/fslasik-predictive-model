import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df_fs = pd.read_csv('results/clean_fs.csv')

def partial_correlation(x, y, covars):
    data = pd.concat([x, y, covars], axis=1).dropna()
    if len(data) < 10:
        return np.nan, np.nan

    x_clean = data[x.name]
    y_clean = data[y.name]
    covars_clean = data[covars.columns]

    mod_x = sm.OLS(x_clean, sm.add_constant(covars_clean)).fit(disp=0)
    res_x = mod_x.resid

    mod_y = sm.OLS(y_clean, sm.add_constant(covars_clean)).fit(disp=0)
    res_y = mod_y.resid

    r, p = stats.pearsonr(res_x, res_y)
    return r, p

def compute_correlation_table(df, dataset_name, confounders=['Age', 'IOP']):
    top_features = pd.read_csv(f'results/ML_Importance_{dataset_name}.csv').head(10)['Feature'].tolist()

    df = df.copy()
    if 'Eye_Right' in top_features and 'Eye_Right' not in df.columns:
        df['Eye_Right'] = df['Eye'].replace({'OD': 'R', 'OS': 'L'}).map({'R': 1, 'L': 0})

    results = []

    for feature in top_features:
        temp_unadj = df[[feature, 'Post_SE']].dropna()
        if len(temp_unadj) > 10:
            r_unadj, p_unadj = stats.pearsonr(temp_unadj[feature], temp_unadj['Post_SE'])
        else:
            r_unadj, p_unadj = np.nan, np.nan

        adj_conf = [c for c in confounders if c != feature]
        if len(adj_conf) > 0:
            r_adj, p_adj = partial_correlation(df[feature], df['Post_SE'], df[adj_conf])
        else:
            r_adj, p_adj = r_unadj, p_unadj

        results.append({
            'Feature': feature,
            'Unadjusted_Correlation': r_unadj,
            'Unadjusted_P_Value': p_unadj,
            'Adjusted_Correlation': r_adj,
            'Adjusted_P_Value': p_adj
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(f'results/Table3_Correlation_{dataset_name}.csv', index=False)
    print(f"Table 3 Correlation saved for {dataset_name}")

if __name__ == '__main__':
    compute_correlation_table(df_fs, 'FSLASIK')
