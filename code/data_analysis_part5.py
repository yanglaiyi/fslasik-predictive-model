import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_fs = pd.read_csv('results/clean_fs.csv')

def run_change_in_estimate(df, dataset_name, key_factors, confounders=['Age', 'IOP', 'Pachy_Thin_Locat']):
    df = df.copy()
    if 'Eye_Right' in key_factors and 'Eye_Right' not in df.columns and 'Eye' in df.columns:
        df['Eye_Right'] = df['Eye'].replace({'OD': 'R', 'OS': 'L'}).map({'R': 1, 'L': 0})
    rows = []
    for key_factor in key_factors:
        adj_conf = [c for c in confounders if c != key_factor]
        needed = [key_factor, 'Outcome_Binary'] + adj_conf
        temp = df[needed].dropna().copy()
        if len(temp) < 20:
            rows.append({
                'Dataset': dataset_name,
                'Key_Factor': key_factor,
                'Crude_Beta': np.nan,
                'Adjusted_Beta': np.nan,
                'Change_Percent': np.nan,
                'Confounding_Validated_gt15': False
            })
            continue
        y = temp['Outcome_Binary']
        try:
            crude_model = sm.Logit(y, sm.add_constant(temp[[key_factor]])).fit(disp=0)
            crude_beta = crude_model.params[key_factor]
            adj_model = sm.Logit(y, sm.add_constant(temp[[key_factor] + adj_conf])).fit(disp=0)
            adj_beta = adj_model.params[key_factor]
            if crude_beta == 0:
                change_pct = np.nan
                validated = False
            else:
                change_pct = abs((adj_beta - crude_beta) / crude_beta) * 100
                validated = change_pct > 15
        except:
            crude_beta, adj_beta, change_pct, validated = np.nan, np.nan, np.nan, False
        rows.append({
            'Dataset': dataset_name,
            'Key_Factor': key_factor,
            'Crude_Beta': crude_beta,
            'Adjusted_Beta': adj_beta,
            'Change_Percent': change_pct,
            'Confounding_Validated_gt15': validated
        })
    pd.DataFrame(rows).to_csv(f'results/ChangeInEstimate_{dataset_name}.csv', index=False)

def get_top_features(dataset_name):
    df = pd.read_csv(f'results/ML_Importance_{dataset_name}.csv')
    return df.head(10)['Feature'].tolist()

def calculate_cutoff(df, feature, confounders=['Age', 'IOP', 'Pachy_Thin_Locat']):
    df = df.copy()
    if feature == 'Eye_Right' and 'Eye_Right' not in df.columns and 'Eye' in df.columns:
        df['Eye_Right'] = df['Eye'].replace({'OD': 'R', 'OS': 'L'}).map({'R': 1, 'L': 0})
    adj_conf = [c for c in confounders if c != feature]
    cols = list(set([feature, 'Outcome_Binary'] + confounders))
    temp = df[cols].dropna()
    y = temp['Outcome_Binary']
    X_unadj = temp[[feature]]

    logit_unadj = sm.Logit(y, sm.add_constant(X_unadj)).fit(disp=0)
    fpr, tpr, thresholds = roc_curve(y, logit_unadj.predict())
    youden = tpr - fpr
    opt_idx = np.argmax(youden)
    unadj_cutoff = thresholds[opt_idx]

    or_unadj = np.exp(logit_unadj.params[feature])
    ci_unadj = np.exp(logit_unadj.conf_int().loc[feature])
    p_unadj = logit_unadj.pvalues[feature]

    adj_conf = [c for c in confounders if c != feature]
    cols_adj = [feature] + adj_conf
    X_adj = temp[cols_adj]
    logit_adj = sm.Logit(y, sm.add_constant(X_adj)).fit(disp=0)
    fpr, tpr, thresholds = roc_curve(y, logit_adj.predict())
    youden = tpr - fpr
    opt_idx = np.argmax(youden)
    adj_cutoff = thresholds[opt_idx]

    or_adj = np.exp(logit_adj.params[feature])
    ci_adj = np.exp(logit_adj.conf_int().loc[feature])
    p_adj = logit_adj.pvalues[feature]

    unadj_thresh = None
    best_j_unadj = -1
    for t in temp[feature].unique():
        pred_class = (temp[feature] >= t).astype(int)
        if logit_unadj.params[feature] < 0:
            pred_class = (temp[feature] <= t).astype(int)
        tpr_t = (pred_class & y).sum() / max(y.sum(), 1)
        fpr_t = (pred_class & (1-y)).sum() / max((1-y).sum(), 1)
        j = tpr_t - fpr_t
        if j > best_j_unadj:
            best_j_unadj = j
            unadj_thresh = t

    adj_thresh = None
    best_j_adj = -1
    for t in temp[feature].unique():
        pred_class = (temp[feature] >= t).astype(int)
        if logit_adj.params[feature] < 0:
            pred_class = (temp[feature] <= t).astype(int)
        tpr_t = (pred_class & y).sum() / max(y.sum(), 1)
        fpr_t = (pred_class & (1-y)).sum() / max((1-y).sum(), 1)
        j = tpr_t - fpr_t
        if j > best_j_adj:
            best_j_adj = j
            adj_thresh = t

    median_val = temp[feature].median()
    temp['Exposure'] = (temp[feature] >= median_val).astype(int)

    try:
        ps_model = sm.Logit(temp['Exposure'], sm.add_constant(temp[adj_conf])).fit(disp=0)
        temp['PS'] = ps_model.predict()

        temp['PS'] = temp['PS'].clip(0.01, 0.99)
        temp['IPTW'] = np.where(temp['Exposure'] == 1, 1/temp['PS'], 1/(1-temp['PS']))

        weighted_model = sm.GLM(temp['Outcome_Binary'], sm.add_constant(temp[[feature]]),
                                family=sm.families.Binomial(),
                                var_weights=temp['IPTW']).fit()

        fpr_w, tpr_w, thresholds_w = roc_curve(temp['Outcome_Binary'], weighted_model.predict(), sample_weight=temp['IPTW'])
        youden_w = tpr_w - fpr_w
        opt_idx_w = np.argmax(youden_w)
        ipw_cutoff = thresholds_w[opt_idx_w]

        best_thresh_w = None
        best_j_w = -1
        for t in temp[feature].unique():
            pred_class = (temp[feature] >= t).astype(int)
            if weighted_model.params[feature] < 0:
                pred_class = (temp[feature] <= t).astype(int)

            true_pos_w = (pred_class * temp['Outcome_Binary'] * temp['IPTW']).sum()
            actual_pos_w = (temp['Outcome_Binary'] * temp['IPTW']).sum()
            false_pos_w = (pred_class * (1 - temp['Outcome_Binary']) * temp['IPTW']).sum()
            actual_neg_w = ((1 - temp['Outcome_Binary']) * temp['IPTW']).sum()

            tpr_t = true_pos_w / max(actual_pos_w, 1e-5)
            fpr_t = false_pos_w / max(actual_neg_w, 1e-5)
            j = tpr_t - fpr_t
            if j > best_j_w:
                best_j_w = j
                best_thresh_w = t

        ipw_or = np.exp(weighted_model.params[feature])
        ipw_ci = np.exp(weighted_model.conf_int().loc[feature])
        ipw_p = weighted_model.pvalues[feature]
    except Exception as e:
        print(f"IPW failed for {feature}: {e}")
        best_thresh_w = np.nan
        ipw_or = np.nan
        ipw_ci = [np.nan, np.nan]
        ipw_p = np.nan

    return {
        'Feature': feature,
        'Unadj_Cutoff': unadj_thresh,
        'Unadj_OR': f"{or_unadj:.2f} ({ci_unadj[0]:.2f}-{ci_unadj[1]:.2f})",
        'Unadj_P': p_unadj,
        'Adj_Cutoff': adj_thresh,
        'Adj_OR': f"{or_adj:.2f} ({ci_adj[0]:.2f}-{ci_adj[1]:.2f})",
        'Adj_P': p_adj,
        'IPW_Cutoff': best_thresh_w,
        'IPW_OR': f"{ipw_or:.2f} ({ipw_ci[0]:.2f}-{ipw_ci[1]:.2f})" if not pd.isna(ipw_or) else "NA",
        'IPW_P': ipw_p
    }

def run_cutoffs(df, dataset_name):
    features = get_top_features(dataset_name)
    results = []
    for f in features:
        try:
            res = calculate_cutoff(df, f)
            results.append(res)

            plt.figure(figsize=(6, 5))
            cutoff = res['Adj_Cutoff']
            group_col = f'{f}_Group'
            df[group_col] = np.where(df[f] >= cutoff, f'>= {cutoff:.2f}', f'< {cutoff:.2f}')
            sns.boxplot(data=df, x=group_col, y='Post_SE', palette=['#4C72B0', '#DD8452'])
            sns.stripplot(data=df, x=group_col, y='Post_SE', color='black', alpha=0.3)
            plt.title(f'{f} Cutoff Analysis - {dataset_name}')
            plt.tight_layout()
            plt.savefig(f'figures/Cutoff_{f}_{dataset_name}.pdf')
            plt.close()
        except Exception as e:
            print(f"Failed cutoff for {f}: {e}")

    pd.DataFrame(results).to_csv(f'results/Cutoffs_{dataset_name}.csv', index=False)

def icc_a1_from_matrix(mat):
    n, k = mat.shape
    grand = mat.values.mean()
    subj_mean = mat.mean(axis=1)
    rater_mean = mat.mean(axis=0)
    msr = k * np.var(subj_mean, ddof=1)
    msc = n * np.var(rater_mean, ddof=1)
    resid = mat.values - subj_mean.values[:, None] - rater_mean.values[None, :] + grand
    mse = np.sum(resid ** 2) / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    f_stat = msr / mse if mse > 0 else np.inf
    p_val = 1 - stats.f.cdf(f_stat, n - 1, (n - 1) * (k - 1))
    return icc, p_val

def calculate_icc(df, dataset_name):
    df = df.copy()

    p_id = 0
    true_ids = []
    for i in range(len(df)):
        if i == 0:
            p_id += 1
        else:
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            if prev['Age'] == curr['Age'] and prev['Gender'] == curr['Gender'] and prev['Eye'] in ['R', 'OD'] and curr['Eye'] in ['L', 'OS']:
                pass
            else:
                p_id += 1
        true_ids.append(p_id)
    df['Patient_ID'] = true_ids

    df['Eye'] = df['Eye'].replace({'OD': 'R', 'OS': 'L'})
    counts = df['Patient_ID'].value_counts()
    both_eyes = counts[counts == 2].index
    df_both = df[df['Patient_ID'].isin(both_eyes)]

    if len(df_both) == 0: return

    pivoted = df_both.pivot(index='Patient_ID', columns='Eye', values='Post_SE')
    if len(pivoted.columns) == 2:
        c1, c2 = pivoted.columns
        pivoted = pivoted.dropna()
        if len(pivoted) < 3:
            return
        icc, p_val = icc_a1_from_matrix(pivoted[[c1, c2]])

        boot_vals = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            idx = rng.integers(0, len(pivoted), len(pivoted))
            boot = pivoted.iloc[idx]
            try:
                b_icc, _ = icc_a1_from_matrix(boot[[c1, c2]])
                if np.isfinite(b_icc):
                    boot_vals.append(b_icc)
            except:
                pass
        if len(boot_vals) > 20:
            ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
        else:
            ci_low, ci_high = np.nan, np.nan

        plt.figure(figsize=(6,6))
        plt.scatter(pivoted[c1], pivoted[c2], alpha=0.6, color='#4C72B0')

        lims = [
            np.min([plt.xlim()[0], plt.ylim()[0]]),
            np.max([plt.xlim()[1], plt.ylim()[1]])
        ]
        plt.plot(lims, lims, 'k--', alpha=0.75)
        plt.xlabel(f'{c1} Eye Post_SE')
        plt.ylabel(f'{c2} Eye Post_SE')
        plt.title(f'Binocular Correlation - {dataset_name}\nICC={icc:.3f} (95%CI {ci_low:.3f}-{ci_high:.3f}), p={p_val:.3f}')
        plt.tight_layout()
        plt.savefig(f'figures/ICC_{dataset_name}.pdf')
        plt.close()

run_cutoffs(df_fs, 'FSLASIK')
calculate_icc(df_fs, 'FSLASIK')
run_change_in_estimate(df_fs, 'FSLASIK', get_top_features('FSLASIK'))
print("Cutoffs and ICC generated for FSLASIK.")
