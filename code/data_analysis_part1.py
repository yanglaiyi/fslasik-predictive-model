import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_curve
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

col_names = [
    'Patient_ID', 'Gender', 'Age', 'Eye', 'Myopia_Duration', 'Glasses_Duration',
    'UCVA_Decimal', 'UCVA_Snellen', 'IOP', 'Pupil_Light', 'Pupil_Dark',
    'Endothelial_Cell', 'Pre_Spherical', 'Pre_Cylindrical', 'Pre_SE',
    'Pre_BCVA_Decimal', 'Pre_BCVA_Snellen', 'AL', 'LT', 'WTW', 'CCT_Diff',
    'Front_Rf', 'Front_Rs', 'Front_Rm', 'Front_Rper', 'Front_K1', 'Front_K2',
    'Front_Km', 'Front_Astig', 'Front_Rmin', 'Back_Rf', 'Back_Rs', 'Back_Rm',
    'Back_Rper', 'Back_K1', 'Back_K2', 'Back_Km', 'Back_Astig', 'Back_Rmin',
    'Pupil_Center', 'Pachy_Vertex', 'Cornea_Volume', 'Chamber_Volume', 'ACD',
    'PD', 'Kmax', 'Pachy_Thin_Locat', 'Dist_Vertex_Thin', 'Progression_Min',
    'Progression_Max', 'Progression_Avg', 'ART_max', 'Df', 'Db', 'Dp', 'Dt',
    'Da', 'D', 'OZ', 'TZ', 'TAZ', 'Ablation_Depth_Max', 'Ablation_Depth_Center',
    'Ablation_Depth_Min', 'RST', 'Ablation_Volume', 'Post_UCVA_Decimal',
    'Post_UCVA_Snellen', 'Post_Spherical', 'Post_Cylindrical', 'Post_SE'
]

def clean_data(filepath):
    df = pd.read_excel(filepath)
    df = df.iloc[:, :71]
    df.columns = col_names

    df['Outcome_Group'] = np.where(df['Post_SE'] > -0.5, 'Good', 'Poor')
    df['Outcome_Binary'] = np.where(df['Post_SE'] > -0.5, 1, 0)

    def to_logmar(val):
        if pd.isna(val): return np.nan
        try:
            val = float(val)
            if val <= 0: return np.nan
            return -np.log10(val)
        except:
            return np.nan

    df['Pre_UCVA_LogMAR'] = df['UCVA_Decimal'].apply(to_logmar)
    df['Pre_BCVA_LogMAR'] = df['Pre_BCVA_Decimal'].apply(to_logmar)
    df['Post_LogMAR'] = df['Post_UCVA_Decimal'].apply(to_logmar)

    df['Gender'] = df['Gender'].map({1: 'Male', 2: 'Female', 0: 'Female'})

    for col in df.columns:
        if col not in ['Patient_ID', 'Gender', 'Eye', 'UCVA_Snellen', 'Pre_BCVA_Snellen', 'Post_UCVA_Snellen', 'Outcome_Group']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Post_SE'])
    return df

df_fs = clean_data('source/表格收集-半飞 -3.15.xlsx')

print(f"FS-LASIK shape: {df_fs.shape}")

df_fs.to_csv('results/clean_fs.csv', index=False)
