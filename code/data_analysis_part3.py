import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

df_fs = pd.read_csv('results/clean_fs.csv')

def decimal_to_lines(dec):
    logmar = -np.log10(dec)
    return logmar / 0.1

def plot_figures(df, dataset_name):
    def categorize_acuity(val):
        if pd.isna(val): return np.nan
        if val < 0.5: return '<20/40'
        elif val < 0.66: return '20/40'
        elif val < 0.8: return '20/30'
        elif val < 1.0: return '20/25'
        else: return '>=20/20'

    df['Post_UCVA_Cat'] = df['Post_UCVA_Decimal'].apply(categorize_acuity)
    df['Pre_BCVA_Cat'] = df['Pre_BCVA_Decimal'].apply(categorize_acuity)

    cats = ['<20/40', '20/40', '20/30', '20/25', '>=20/20']
    post_counts = df['Post_UCVA_Cat'].value_counts(normalize=True).reindex(cats).fillna(0) * 100
    pre_counts = df['Pre_BCVA_Cat'].value_counts(normalize=True).reindex(cats).fillna(0) * 100

    post_cum = post_counts[::-1].cumsum()[::-1]
    pre_cum = pre_counts[::-1].cumsum()[::-1]

    acuity_df = pd.DataFrame({'Preoperative BCVA': pre_cum, 'Postoperative UCVA': post_cum}).reset_index()
    acuity_df = acuity_df.melt(id_vars='index', var_name='Type', value_name='Cumulative percentage (%)')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=acuity_df, x='index', y='Cumulative percentage (%)', hue='Type', palette=['#4C72B0', '#DD8452'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.xlabel('Visual Acuity (Snellen)')
    plt.ylabel('Cumulative percentage (%)')
    plt.title(f'Cumulative Visual Acuity - {dataset_name}')
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig(f'figures/Fig2A_{dataset_name}.pdf')
    plt.close()

    def calc_line_diff(post, pre):
        if pd.isna(post) or pd.isna(pre) or post <= 0 or pre <= 0: return np.nan
        post_log = -np.log10(post)
        pre_log = -np.log10(pre)
        return round((pre_log - post_log) / 0.1)

    df['Line_Change'] = df.apply(lambda row: calc_line_diff(row['Post_UCVA_Decimal'], row['Pre_BCVA_Decimal']), axis=1)
    line_counts = df['Line_Change'].value_counts(normalize=True).sort_index() * 100

    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(x=line_counts.index, y=line_counts.values, color='#4C72B0')
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.xlabel('Change in Snellen lines')
    plt.ylabel('Eyes (%)')
    plt.title(f'Change in Visual Acuity Lines - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'figures/Fig2B_{dataset_name}.pdf')
    plt.close()

    plt.figure(figsize=(8, 6))
    bins = [-np.inf, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, np.inf]
    labels = ['<= -2.0', '(-2.0, -1.5]', '(-1.5, -1.0]', '(-1.0, -0.5]', '(-0.5, 0]', '(0, 0.5]', '(0.5, 1.0]', '> 1.0']

    se_cuts = pd.cut(df['Post_SE'], bins=bins, labels=labels, right=True, include_lowest=False)
    se_counts = se_cuts.value_counts(normalize=True).sort_index() * 100

    ax3 = sns.barplot(x=se_counts.index, y=se_counts.values, color='#4C72B0')
    for p in ax3.patches:
        ax3.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Postoperative SE (D)')
    plt.ylabel('Eyes (%)')
    plt.title(f'Postoperative SE Distribution - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'figures/Fig2C_{dataset_name}.pdf')
    plt.close()

    plt.figure(figsize=(10, 6))
    bins_cyl = [-np.inf, 0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    labels_cyl = ['0', '0.1-0.5', '0.6-1.0', '1.1-1.5', '1.6-2.0', '2.1-2.5', '2.6-3.0', '>3.0']

    pre_cyl = pd.cut(df['Pre_Cylindrical'].abs(), bins=bins_cyl, labels=labels_cyl).value_counts(normalize=True).sort_index() * 100
    post_cyl = pd.cut(df['Post_Cylindrical'].abs(), bins=bins_cyl, labels=labels_cyl).value_counts(normalize=True).sort_index() * 100

    cyl_df = pd.DataFrame({'Preoperative': pre_cyl, 'Postoperative': post_cyl}).reset_index()
    cyl_df = cyl_df.melt(id_vars='index', var_name='Time', value_name='Percentage')

    ax4 = sns.barplot(data=cyl_df, x='index', y='Percentage', hue='Time', palette=['#4C72B0', '#DD8452'])
    for p in ax4.patches:
        ax4.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    plt.xlabel('Cylinder (D)')
    plt.ylabel('Eyes (%)')
    plt.title(f'Cylinder Distribution - {dataset_name}')
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig(f'figures/Fig2D_{dataset_name}.pdf')
    plt.close()

plot_figures(df_fs, 'FSLASIK')
print("Figures 2A-2D generated for FSLASIK.")
