# 🌸 Iris Dataset — Exploratory Data Analysis (EDA)

> Univariate distributions · Skewness profiling · Bivariate analysis · Outlier detection · Correlation analysis

---

## 📋 Overview

This notebook performs a comprehensive **Exploratory Data Analysis (EDA)** on the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) — a benchmark collection of 150 observations describing morphological measurements of three iris species: *Setosa*, *Versicolor*, and *Virginica*.

---

## 📁 Project Structure

```
├── iriseda.ipynb       # Main analysis notebook
├── iris.csv            # Dataset
└── README.md           # This file
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 150 |
| Features | 4 numeric + 1 categorical |
| Target | `variety` (Setosa, Versicolor, Virginica) |
| Missing values | None |
| Duplicate rows | 1 |

| Feature | Type | Description |
|---|---|---|
| `sepal.length` | float64 | Length of the sepal (cm) |
| `sepal.width` | float64 | Width of the sepal (cm) |
| `petal.length` | float64 | Length of the petal (cm) |
| `petal.width` | float64 | Width of the petal (cm) |
| `variety` | object | Species label |

---

## 🔬 Analysis Steps

### 1. Data Loading & Quality Check
- Shape inspection and `dtypes` verification
- Missing value check → **no missing values found**
- Duplicate detection → **1 duplicate row identified**

### 2. Univariate Analysis — Feature Distributions
- Overlaid histograms and KDE curves per species for all four features
- Per-species mean lines as vertical dashed markers
- Custom colour palette: Setosa `#4C9BE8` · Versicolor `#F4845F` · Virginica `#57B894`

### 3. Skewness Detection
- Skewness computed globally for each feature using `scipy.stats.skew`
- Colour-coded by direction: right-skewed (red) · symmetric (green) · left-skewed (blue)
- Mean and median lines overlaid for visual confirmation

### 4. Bivariate Analysis — Boxplots
- Boxplots with jittered strip overlays per species and feature
- **Kruskal-Wallis H-test** applied to assess statistical significance of inter-species differences

### 5. Outlier Detection
- **IQR method**: lower fence = Q1 − 1.5×IQR · upper fence = Q3 + 1.5×IQR
- **Z-score method**: threshold |z| > 3
- Results visualised with colour-coded scatter markers

### 6. Correlation Analysis
- Pearson correlation matrix for all numeric features
- Full heatmap + per-variety heatmaps + lower-triangle heatmap
- Correlation strength annotations (Weak / Moderate / Strong / Very Strong)

---

## 📌 Key Findings

| Finding | Detail |
|---|---|
| Data quality | No missing values · 1 duplicate row |
| Best species separator | Petal features (Setosa clearly distinct) |
| Skewness | All 4 features symmetric (\|skew\| < 0.5) |
| Outliers | Only `sepal.width` — 4 (IQR) · 1 (Z-score) |
| Strongest correlation | `petal.length` ↔ `petal.width` — r = +0.96 |
| Statistical significance | All inter-species differences p < 0.001 (Kruskal-Wallis) |

---

## 🛠️ Requirements

```bash
pip install pandas matplotlib scipy numpy
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `matplotlib` | Visualisations |
| `scipy` | KDE, skewness, Kruskal-Wallis |
| `numpy` | Numerical operations |

---

## ▶️ How to Run

```bash
git clone <repo-url>
cd <repo-folder>
jupyter notebook iriseda.ipynb
```

> **Note:** The notebook was originally developed with a **Python (Pyodide)** kernel. Any standard Python 3 environment with the packages above is sufficient for local execution.

---

## 📝 Abstract

This notebook presents a structured EDA of the Iris dataset. Analysis progresses through data quality assessment, univariate and bivariate visualisation, skewness profiling, outlier detection, and correlation analysis. All four features exhibit symmetric distributions at the dataset level. Petal measurements are the strongest discriminators between species, confirmed by highly significant Kruskal-Wallis tests (p < 0.001). A near-perfect correlation (r = 0.96) between petal length and petal width indicates high multicollinearity, relevant for any subsequent modelling step.

---

## ✍️ Written Observations

**1. Data Quality is High, with One Notable Anomaly**
The dataset contains no missing values across any of its columns. However, one duplicate row was identified during quality inspection. While its impact is negligible given the dataset's size, it is a relevant data integrity flag that should be removed before any downstream modelling task.

**2. Petal Features are the Strongest Discriminators Between Species**
Petal length and petal width exhibit well-separated, non-overlapping distributions for *Setosa* compared to the other two species, while *Versicolor* and *Virginica* show moderate overlap. Sepal features display considerably more overlap across all three varieties. This visual evidence is reinforced by the Kruskal-Wallis H-test, where all four features returned highly significant results (p < 0.001).

**3. All Features Exhibit Symmetric Distributions at the Dataset Level**
Skewness values fall within [−0.5, +0.5] for all features: sepal length (+0.31), sepal width (+0.32), petal length (−0.27), and petal width (−0.10). Mean and median values are correspondingly close for each feature, confirming the absence of heavy tail distortions — a favourable property for most statistical and machine learning assumptions.

**4. Sepal Width is the Only Feature Containing Outliers**
Both the IQR method and the Z-score method consistently identify `sepal.width` as the sole feature with anomalous values. The IQR method flags four observations (rows 15, 32, 33, 60), while the stricter Z-score threshold flags only row 15. This concentration of anomalies in a single feature suggests a structural characteristic of sepal width's natural variability, particularly within the *Setosa* species.

**5. Petal Length and Petal Width Share a Near-Perfect Correlation**
The Pearson correlation matrix reveals that petal length and petal width are very strongly correlated (r = +0.96). Sepal length also correlates strongly with both petal features (r = +0.87 and r = +0.82 respectively), while sepal width is negatively and weakly correlated with the remaining features. This multicollinearity suggests that dimensionality reduction (e.g. PCA) or careful feature selection should be considered before predictive modelling.
