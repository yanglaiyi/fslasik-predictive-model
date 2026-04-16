import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_fs = pd.read_csv('results/clean_fs.csv')

table1_vars = [
    'Gender', 'Age', 'Eye', 'Myopia_Duration', 'Glasses_Duration',
    'UCVA_Decimal', 'IOP', 'Pupil_Light', 'Pupil_Dark', 'Endothelial_Cell',
    'Pre_Spherical', 'Pre_Cylindrical', 'Pre_SE', 'Pre_BCVA_Decimal',
    'AL', 'LT', 'WTW', 'Front_Rf', 'Front_Rs', 'Front_Rm', 'Front_Rper',
    'Front_K1', 'Front_K2', 'Front_Km', 'Front_Astig', 'Front_Rmin',
    'Back_Rf', 'Back_Rs', 'Back_Rm', 'Back_Rper', 'Back_K1', 'Back_K2',
    'Back_Km', 'Back_Astig', 'Back_Rmin', 'Pupil_Center', 'Pachy_Vertex',
    'Cornea_Volume', 'Chamber_Volume', 'ACD', 'PD', 'Kmax', 'Pachy_Thin_Locat',
    'Dist_Vertex_Thin', 'Progression_Min', 'Progression_Max', 'Progression_Avg',
    'ART_max', 'Df', 'Db', 'Dp', 'Dt', 'Da', 'D', 'OZ', 'TZ', 'TAZ',
    'Ablation_Depth_Max', 'Ablation_Depth_Center', 'Ablation_Depth_Min', 'RST',
    'Ablation_Volume', 'Post_UCVA_Decimal', 'Post_Spherical', 'Post_Cylindrical', 'Post_SE'
]

def generate_table1(df, dataset_name):
    results = []
    g1 = df[df['Outcome_Group'] == 'Good']
    g2 = df[df['Outcome_Group'] == 'Poor']

    for var in table1_vars:
        if var not in df.columns: continue

        if df[var].dtype == 'O' or df[var].nunique() <= 2:
            counts = df[var].value_counts()

            crosstab = pd.crosstab(df[var], df['Outcome_Group'])
            try:
                chi2, p, _, _ = stats.chi2_contingency(crosstab, correction=False)
                if crosstab.min().min() < 5:
                    if crosstab.shape == (2, 2):
                        _, p = stats.fisher_exact(crosstab)
            except:
                p = np.nan

            for val in counts.index:
                n_all = (df[var] == val).sum()
                n_g1 = (g1[var] == val).sum()
                n_g2 = (g2[var] == val).sum()
                pct_all = n_all / len(df) * 100
                pct_g1 = n_g1 / len(g1) * 100 if len(g1)>0 else 0
                pct_g2 = n_g2 / len(g2) * 100 if len(g2)>0 else 0

                results.append({
                    'Variable': f"{var} ({val})",
                    'Total (n={})'.format(len(df)): f"{n_all} ({pct_all:.1f}%)",
                    'Good (n={})'.format(len(g1)): f"{n_g1} ({pct_g1:.1f}%)",
                    'Poor (n={})'.format(len(g2)): f"{n_g2} ({pct_g2:.1f}%)",
                    'P-value': f"{p:.4f}" if val == counts.index[0] else ""
                })
        else:
            _, p_sw = stats.shapiro(df[var].dropna())
            is_normal = p_sw > 0.05

            if is_normal:
                mean_all, std_all = df[var].mean(), df[var].std()
                mean_g1, std_g1 = g1[var].mean(), g1[var].std()
                mean_g2, std_g2 = g2[var].mean(), g2[var].std()

                if len(g1[var].dropna()) > 0 and len(g2[var].dropna()) > 0:
                    _, p = stats.ttest_ind(g1[var].dropna(), g2[var].dropna(), equal_var=False)
                else:
                    p = np.nan

                results.append({
                    'Variable': var,
                    'Total (n={})'.format(len(df)): f"{mean_all:.2f} ± {std_all:.2f}",
                    'Good (n={})'.format(len(g1)): f"{mean_g1:.2f} ± {std_g1:.2f}",
                    'Poor (n={})'.format(len(g2)): f"{mean_g2:.2f} ± {std_g2:.2f}",
                    'P-value': f"{p:.4f}" if not pd.isna(p) else "NA"
                })
            else:
                med_all, iqr_all = df[var].median(), df[var].quantile(0.75) - df[var].quantile(0.25)
                med_g1, iqr_g1 = g1[var].median(), g1[var].quantile(0.75) - g1[var].quantile(0.25)
                med_g2, iqr_g2 = g2[var].median(), g2[var].quantile(0.75) - g2[var].quantile(0.25)

                if len(g1[var].dropna()) > 0 and len(g2[var].dropna()) > 0:
                    _, p = stats.mannwhitneyu(g1[var].dropna(), g2[var].dropna())
                else:
                    p = np.nan

                results.append({
                    'Variable': var,
                    'Total (n={})'.format(len(df)): f"{med_all:.2f} ({iqr_all:.2f})",
                    'Good (n={})'.format(len(g1)): f"{med_g1:.2f} ({iqr_g1:.2f})",
                    'Poor (n={})'.format(len(g2)): f"{med_g2:.2f} ({iqr_g2:.2f})",
                    'P-value': f"{p:.4f}" if not pd.isna(p) else "NA"
                })

    pd.DataFrame(results).to_csv(f'results/Table1_{dataset_name}.csv', index=False)

generate_table1(df_fs, 'FSLASIK')
print("Table 1 generated for FSLASIK.")
