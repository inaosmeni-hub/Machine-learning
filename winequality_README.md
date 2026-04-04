# 🍷 Wine Quality Dataset — Data Preprocessing & Feature Engineering

> Missing values · Duplicate removal · Outlier detection · Feature scaling · Feature engineering · Variance filtering · Correlation analysis

---

## 📋 Overview

This notebook implements a complete **data preprocessing and feature engineering pipeline** on the [WineQT dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) — a collection of 1,143 physicochemical measurements of red wine samples labelled with expert quality scores (3–8). The goal is to produce a clean, enriched, and model-ready feature set for downstream quality prediction tasks.

---

## 📁 Project Structure

```
├── winequality__1_.ipynb       # Main preprocessing notebook
├── WineQT.csv                  # Dataset
├── FEATURE_ENGINEERING.md      # Auto-generated feature summary
└── README.md                   # This file
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | `WineQT.csv` |
| Rows (raw) | 1,143 |
| Rows after deduplication | 1,018 |
| Original features | 11 numeric + 1 target |
| Target variable | `quality` (integer, 3–8) |
| Dropped column | `Id` (non-informative) |

| Feature | Type | Description |
|---|---|---|
| `fixed acidity` | float64 | Non-volatile acids |
| `volatile acidity` | float64 | Acetic acid — high values cause vinegar taste |
| `citric acid` | float64 | Adds freshness and flavour |
| `residual sugar` | float64 | Sugar remaining after fermentation |
| `chlorides` | float64 | Salt content |
| `free sulfur dioxide` | float64 | Active SO₂ preservative |
| `total sulfur dioxide` | float64 | Total SO₂ (free + bound) |
| `density` | float64 | Mass per unit volume |
| `pH` | float64 | Acidity level |
| `sulphates` | float64 | Contributes to SO₂ levels |
| `alcohol` | float64 | Alcohol content (% vol) |
| `quality` | int64 | Expert quality score (target) |

---

## 🔬 Pipeline Steps

### 1. Missing Value Report & Handling
- Column-by-column audit with counts and percentages
- Strategy: median imputation for numeric · mode for categorical
- Result: **no missing values found** — no imputation applied

### 2. Duplicate Detection & Removal
- **125 duplicate rows detected** (10.94% of dataset)
- Removed with `keep='first'`
- Final shape after deduplication: **(1,018 × 12)**

### 3. Outlier Detection & Treatment
- Detection via **IQR rule** (1.5× fence) and **Z-score** (|z| > 3)
- Consensus flagging: rows flagged by both methods
- Most affected: `residual sugar` (110) · `chlorides` (77) · `fixed acidity` (44)
- Treatment: **IQR Winsorisation** (clipping) — 415 values capped, all rows preserved

### 4. Categorical Variable Check
- All 12 columns confirmed numeric — **no encoding required**

### 5. Feature Scaling
- Three scalers compared: `StandardScaler` · `MinMaxScaler` · `RobustScaler`
- **StandardScaler** selected as primary (data approximately normal post-Winsorisation)
- Before/after distributions visualised with paired KDE plots

### 6. Train / Test Split
- Split: **80% train / 20% test**
- **Stratified** by `quality` to preserve class proportions
- Random state: 42
- Shapes: `X_train (914, 11)` · `X_test (229, 11)`

### 7. Feature Engineering — Ratio & Interaction Features

| Feature | Formula | Interpretation |
|---|---|---|
| `so2_ratio` | `free SO₂ / total SO₂` | Active preservation ratio |
| `total_acidity` | `fixed + volatile + citric acid` | Overall acid load |
| `acid_ratio` | `fixed acidity / volatile acidity` | Acid quality balance |
| `alcohol_density` | `alcohol / density` | Body and strength index |
| `sulphates_chlorides` | `sulphates / chlorides` | Preservation vs saltiness |

### 8. Binning & One-Hot Encoding

| Feature | Bins | Thresholds |
|---|---|---|
| `alcohol_bin` | Low · Medium · High | 9.5 · 11.1 |
| `sulphates_bin` | Low · Medium · High | 0.55 · 0.73 |
| `volatile_acidity_bin` | Low · Medium · High | 0.39 · 0.64 |

### 9. Variance Threshold Filtering
- Threshold: **0.01**
- Dropped: `density` · `chlorides`
- Retained: **14 features**

### 10. Correlation with Target
- Pearson r computed for all features against `quality`
- Strongest positive: `alcohol` (r = +0.489) · `alcohol_density` (r = +0.488)
- Strongest negative: `volatile acidity` (r = −0.405)
- Weakest: `residual sugar` (r = +0.034) — flagged for potential removal

---

## 📌 Key Findings

| Finding | Detail |
|---|---|
| Data quality | No missing values · 125 duplicates (10.94%) removed |
| Outlier treatment | Winsorisation — 415 values capped · 0 rows lost |
| Best original predictor | `alcohol` (r = +0.489) · `volatile acidity` (r = −0.405) |
| Best engineered feature | `sulphate_alcohol` (r = +0.496) |
| Eliminated by variance | `density` · `chlorides` |
| Class imbalance | Quality 5–6 = ~83% of all samples |

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisations |
| `scipy` | Statistical tests, Z-score |
| `scikit-learn` | Scalers, train/test split, variance threshold |

---

## ▶️ How to Run

```bash
git clone <repo-url>
cd <repo-folder>
jupyter notebook winequality__1_.ipynb
```

> **Note:** The notebook was developed with a **Python (Pyodide)** kernel. Any standard Python 3 environment with the packages above is sufficient for local execution.

---

## 📝 Abstract

This notebook implements a structured end-to-end data preprocessing and feature engineering pipeline on the WineQT dataset. The pipeline addresses data quality issues including duplicates and outliers, establishes a clean and scaled feature space, and constructs new domain-informed features to enhance predictive signal. Each step is documented with detailed reports and visualisations, producing a model-ready dataset of 14 features derived from the original 11.

---

## ✍️ Written Observations

**1. Duplicate Rows Represent the Most Significant Data Quality Issue**
While the dataset contains no missing values, it carries 125 exact duplicate rows (10.94% of total). This non-trivial proportion could meaningfully skew training distributions and inflate model performance if left untreated. After removing duplicates while retaining the first occurrence, the working dataset is reduced to 1,018 samples — prioritising data integrity over volume.

**2. Outlier Treatment via Winsorisation Preserves All Rows While Reducing Distributional Distortion**
Outlier detection using both IQR and Z-score methods revealed widespread anomalies across multiple features, with `residual sugar` (110 IQR flags) and `chlorides` (77) most affected. Rather than removing flagged rows — which would further reduce an already moderate dataset — Winsorisation was applied, capping 415 individual values while preserving all 1,143 rows.

**3. Alcohol and Volatile Acidity Are the Strongest Original Predictors of Quality**
Pearson correlation analysis reveals a clear hierarchy: alcohol (r = +0.489) shows the strongest positive relationship with quality, consistent with the principle that higher-alcohol wines tend to score higher. Volatile acidity (r = −0.405) has the strongest negative relationship, reflecting the well-established oenological effect of acetic acid on perceived wine quality. Features such as `residual sugar` (r = +0.034) show near-zero correlation.

**4. Feature Engineering Significantly Amplifies Predictive Signal**
The `sulphate_alcohol` interaction feature (r = +0.496) and `density_alcohol_ratio` (r = −0.489) both outperform the individual correlations of their component variables. This confirms that the relationship between physicochemical properties and wine quality is not purely additive — multiplicative and ratio-based combinations capture synergistic effects that raw measurements alone cannot express.

**5. The Quality Score Distribution Is Heavily Imbalanced, with Implications for Modelling**
Quality scores 5 and 6 together account for approximately 82.7% of all samples, while extreme classes (quality 3 at 0.5%) are very sparsely represented. The train/test split was stratified to preserve these proportions. However, this imbalance means that accuracy alone is a misleading evaluation metric, and techniques such as class weighting, SMOTE oversampling, or ordinal regression framing should be considered in subsequent modelling phases.
