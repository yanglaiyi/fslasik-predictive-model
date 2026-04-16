# FSLASIK Predictive Model

## Project Overview

This repository contains the complete analysis pipeline for developing and validating a predictive model of refractive outcomes following Femtosecond Laser-Assisted In Situ Keratomileusis (FS-LASIK) surgery.

## Study Objective

To identify preoperative predictors of suboptimal refractive outcomes (postoperative spherical equivalent > -0.5 D) after FS-LASIK surgery using machine learning ensemble methods and establish clinically applicable cutoff values.

## Repository Structure

```
fslasik-predictive-model/
├── code/                      # Analysis scripts
│   ├── data_analysis_part1.py  # Data cleaning and preprocessing
│   ├── data_analysis_part2.py # Descriptive statistics (Table 1)
│   ├── data_analysis_part3.py # Visual acuity and refractive outcome figures
│   ├── data_analysis_part4.py # Machine learning feature importance
│   ├── data_analysis_part5.py # Cutoff analysis and ICC calculation
│   ├── data_analysis_part6.py # Correlation analysis (Table 3)
│   └── plot_trees_and_bars.py # Tree visualization and PDP plots
└── README.md
```

## Key Findings

### Top 10 Predictive Features
1. Endothelial Cell
2. Pre-Spherical
3. Pre-SE
4. Df
5. Back-Rf
6. PD
7. IOP
8. Pupil-Dark
9. Pre-UCVA (LogMAR)
10. Ablation Volume

### Optimal Cutoff Values (IPW-Adjusted)
| Feature | IPW Cutoff | IPW OR (95% CI) | IPW P-value |
|---------|------------|-----------------|-------------|
| Endothelial Cell | 3036.8 | 1.00 (1.00-1.00) | <0.001 |
| Pre-Spherical | -6.5 D | 1.34 (1.23-1.47) | <0.001 |
| Pre-SE | -4.625 D | 1.30 (1.19-1.43) | <0.001 |
| Df | -0.02 | 1.44 (1.19-1.73) | <0.001 |
| Back-Rf | 6.47 mm | 3.04 (1.42-6.50) | 0.004 |
| PD | 4.21 mm | 0.65 (0.51-0.83) | <0.001 |
| IOP | 17.4 mmHg | 1.05 (0.99-1.12) | 0.114 |
| Pupil-Dark | 7.2 mm | 0.85 (0.69-1.06) | 0.146 |
| Pre-UCVA (LogMAR) | 0.82 | 0.22 (0.13-0.37) | <0.001 |
| Ablation Volume | 2315.0 μm³ | 1.00 (1.00-1.00) | 0.033 |

### Model Performance
- Ensemble method: Random Forest + XGBoost + LASSO voting
- Feature selection: 100-bootstrapped LASSO frequency + tree-based importance
- Validation: 70/30 train-test split with repeated cross-validation

## Methods Summary

| Analysis | Method |
|----------|--------|
| Descriptive Statistics | Mann-Whitney U / Chi-square / Fisher's exact |
| Feature Selection | LASSO (100 bootstraps) + RF + XGBoost ensemble |
| Tree-based Models | Decision Tree (depth=3) |
| Cutoff Determination | Youden's J statistic on ROC curve + IPW validation |
| Confounding Assessment | Change-in-estimate (>15% rule) with covariates: Age, IOP, Pachy_Thin_Locat |
| Correlation Analysis | Partial correlation adjusted for Age, IOP |
| Nonlinear Interaction | Partial Dependence Display (PDP) for top 5 feature pairs |
| Inter-eye Agreement | Intraclass correlation coefficient (ICC) |

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

## How to Reproduce

```bash
# Install dependencies
pip install pandas numpy scipy statsmodels scikit-learn xgboost matplotlib seaborn openpyxl

# Run analysis pipeline (in code/ directory)
python data_analysis_part1.py  # Data cleaning
python data_analysis_part2.py  # Table 1
python data_analysis_part3.py   # Figures 2A-2D
python data_analysis_part4.py  # ML importance
python data_analysis_part5.py  # Cutoffs & ICC
python data_analysis_part6.py  # Table 3 correlation
python plot_trees_and_bars.py  # Tree visualizations
```

## Data Description

### Outcome Variable
- **Post_SE**: Postoperative spherical equivalent (diopters)
- **Outcome_Group**: Good (SE ≤ -0.5 D) vs Poor (SE > -0.5 D)

### Top 10 Predictors
| Variable | Description |
|----------|-------------|
| Endothelial Cell | Endothelial cell count (cells/mm²) |
| Pre-Spherical | Preoperative spherical refraction (D) |
| Pre-SE | Preoperative spherical equivalent (D) |
| Df | Front corneal shape factor |
| Back-Rf | Back corneal radius of curvature (mm) |
| PD | Pupil diameter under light (mm) |
| IOP | Intraocular pressure (mmHg) |
| Pupil-Dark | Dark pupil diameter (mm) |
| Pre-UCVA (LogMAR) | Preoperative uncorrected visual acuity (LogMAR) |
| Ablation Volume | Total ablation volume (μm³) |

## Limitations

1. **Single-center study**: External validation in independent cohorts is required
2. **Retrospective design**: Potential selection bias; prospective validation recommended
3. **Sample size**: Post-hoc power analysis should be conducted; larger cohorts may improve model stability
4. **Missing data**: Multiple imputation not performed; complete case analysis used
5. **Surgical parameters**: Surgeon experience and laser settings not included as predictors
6. **Follow-up period**: Only immediate postoperative outcomes assessed; long-term stability unknown
7. **Ethnic diversity**: Cohort may not represent diverse populations

## Future Directions

- [ ] External validation in independent cohorts
- [ ] Prospective multi-center study
- [ ] Integration of wavefront aberrations
- [ ] Long-term outcome prediction (>1 year)
- [ ] Custom machine learning models for personalized surgery planning
- [ ] Web-based prediction calculator for clinical use

## Citation

If you use this code or data in your research, please cite:

```
Translating Black-Box Algorithms to Clinical Maps: An IPW-Validated Machine Learning Framework for FS-LASIK Risk Stratification
```

## Contact

Corresponding authors: wukunchao@126.com, 34459356@qq.com
Co-author: zgl19970@126.com

## License

This project is for academic research purposes. All rights reserved.

---

**Note**: Raw data files are not included due to privacy concerns. Researchers wishing to replicate the analysis should contact the authors for data access procedures.
