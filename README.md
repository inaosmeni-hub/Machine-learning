# Machine Learning Projects

This repository contains data analysis and preprocessing notebooks built as part of a machine learning coursework. Each notebook covers a different dataset and focuses on a specific stage of the ML pipeline.

---

## Project 1 — Iris Dataset: Exploratory Data Analysis

**Notebook:** `iriseda.ipynb`  
**Dataset:** `iris.csv` — 150 samples, 4 features, 3 species (Setosa, Versicolor, Virginica)

### What the notebook covers

- Data quality check: missing values and duplicates
- Univariate distributions with histograms and KDE curves per species
- Skewness analysis for each feature
- Bivariate analysis using boxplots and the Kruskal-Wallis test
- Outlier detection using IQR and Z-score methods
- Correlation analysis with Pearson heatmaps (overall and per species)

### Key findings

The dataset has no missing values and one duplicate row. Petal length and petal width are the strongest separators between species — Setosa is clearly distinct from the other two. All four features show symmetric distributions (skewness within ±0.5). The only feature with outliers is `sepal.width`, with 4 flagged by IQR and 1 by Z-score. The strongest correlation in the dataset is between petal length and petal width (r = 0.96), which indicates multicollinearity that should be accounted for in any subsequent modelling.

### Observations

1. The dataset is clean with no missing values. One duplicate row was found, which is a minor integrity issue but should be removed before modelling.

2. Petal features separate species much more clearly than sepal features. Setosa is entirely distinct in both petal length and width distributions. Kruskal-Wallis tests confirm that all four features differ significantly across species (p < 0.001).

3. All features are approximately symmetric at the overall dataset level. Skewness values range from −0.27 to +0.32, well within the symmetric threshold of ±0.5.

4. Sepal width is the only feature with measurable outliers. The IQR method flags rows 15, 32, 33, and 60 — row 15 is also flagged by Z-score, making it the most anomalous point in the dataset.

5. Petal length and petal width are almost perfectly correlated (r = 0.96). Sepal length also correlates strongly with both petal features. This level of multicollinearity suggests that PCA or feature selection would be beneficial before training a model.

### Requirements

```
pandas, matplotlib, scipy, numpy
```

---

## Project 2 — Wine Quality Dataset: Preprocessing & Feature Engineering

**Notebook:** `winequality_(1).ipynb`  
**Dataset:** `WineQT.csv` — 1,143 samples, 11 physicochemical features, quality scores from 3 to 8

### What the notebook covers

- Missing value audit and handling strategy
- Duplicate detection and removal
- Outlier detection (IQR + Z-score) and treatment via Winsorisation
- Categorical variable check and encoding assessment
- Feature scaling comparison: StandardScaler, MinMaxScaler, RobustScaler
- Stratified train/test split (80/20)
- Feature engineering: ratio and interaction features
- Binning of alcohol, sulphates, and volatile acidity with one-hot encoding
- Variance threshold filtering
- Correlation ranking of all features against the quality target

### Key findings

The dataset has no missing values but contains 125 duplicate rows (10.94%), which is the most significant data quality issue. Outliers were treated with Winsorisation rather than row removal to preserve sample size — 415 values were capped across 11 features. Alcohol content is the strongest single predictor of quality (r = +0.489), followed closely by volatile acidity (r = −0.405). The engineered feature `sulphate_alcohol` achieves the highest correlation of all features (r = +0.496). Two features — `density` and `chlorides` — were dropped after failing the variance threshold filter. Quality classes 5 and 6 together represent about 83% of all samples, making class imbalance a key challenge for modelling.

### Observations

1. The largest data quality problem is duplication rather than missingness. With 125 duplicate rows accounting for nearly 11% of the dataset, leaving them in would distort any model trained on this data. After removal, the working dataset contains 1,018 samples.

2. Winsorisation was chosen over row deletion for outlier treatment. Given the moderate dataset size, removing all outlier-flagged rows would have caused a disproportionate loss of data. Clipping to IQR bounds capped 415 values while keeping every row intact.

3. Alcohol and volatile acidity are the most informative original features. Alcohol correlates positively with quality (r = +0.489), consistent with the general tendency for higher-alcohol wines to receive higher scores. Volatile acidity shows the strongest negative relationship (r = −0.405), reflecting the known negative effect of acetic acid on perceived wine quality.

4. Feature engineering improves predictive signal beyond what raw features offer. The interaction between sulphates and alcohol and the density-to-alcohol ratio both outperform their individual components, confirming that quality is influenced by combinations of physicochemical properties, not each feature in isolation.

5. The target variable is heavily imbalanced. Scores 5 and 6 dominate the dataset (~83%), while score 3 appears in fewer than 1% of samples. Stratified splitting was applied to preserve proportions, but this imbalance means that standard accuracy is a misleading metric — class weighting or resampling should be considered in the modelling phase.

### Requirements

```
pandas, numpy, matplotlib, scipy, scikit-learn
```

##Project 3 — Heart Disease Prediction: Supervised Learning & Model Comparison

Notebook: heartfailure.ipynb
Dataset: heart.csv — 918 samples, 11 features, binary target (heart disease: yes/no)
What the notebook covers

##Data loading and quality check

Label encoding of categorical features
Train/test split (70/30)
Training and evaluation of four classifiers: Decision Tree, Linear Regression, Logistic Regression, and Naive Bayes
Hyperparameter tuning for Naive Bayes using GridSearchCV
Model comparison across accuracy, precision, recall, and F1 score
ROC curves and AUC for all models
Confusion matrices and per-class precision/recall analysis
Feature importance (Decision Tree) and coefficient analysis (Logistic Regression)

##Key findings

The dataset is clean with no missing values and a near-balanced target (55.3% positive cases). Logistic Regression achieved the best overall accuracy (86.2%) with an AUC of 0.928. After hyperparameter tuning with GridSearchCV, Naive Bayes improved from 84.2% to 87.3% accuracy and reached the highest AUC of 0.933. Linear Regression, used here as a baseline classifier with a 0.5 threshold, performed competitively but is not appropriate for binary classification tasks. The most important feature for the Decision Tree was ST_Slope, consistent with its clinical relevance as an ECG indicator.
Model results
ModelAccuracyAUC-ROCDecision Tree (max_depth=4)0.83700.9121Linear Regression (threshold=0.5)0.8370—Logistic Regression0.86230.9276Naive Bayes (default)0.84240.9090Naive Bayes (tuned)0.87320.9331
Observations

The dataset is well-balanced and complete, with no missing values and a 55/45 split between positive and negative cases. This avoids the class imbalance problems common in medical datasets and means standard accuracy is a reliable evaluation metric here.
Logistic Regression outperforms the other models in accuracy before tuning, and also produces the best-calibrated probability estimates among the baseline models (AUC = 0.928). Its coefficient analysis shows that ST_Slope, ExerciseAngina, and Oldpeak are the strongest predictors of heart disease, which aligns with established clinical knowledge.
Linear Regression is not suited for binary classification but was included as a baseline. Despite reaching the same accuracy as the Decision Tree when thresholded at 0.5, its R² of 0.44 and the clear non-linearity in its residual plot confirm that the model is not capturing the structure of the problem correctly.
Hyperparameter tuning significantly improved Naive Bayes. Using GridSearchCV across seven values of var_smoothing, the optimal value of 1e-7 raised accuracy from 84.2% to 87.3% and AUC from 0.909 to 0.933 — making it the top-performing model overall after tuning. This highlights how sensitive Gaussian Naive Bayes can be to its smoothing parameter.
All four models show higher precision for the Disease class than for No Disease, meaning they are better at correctly identifying sick patients when they predict disease than at identifying healthy patients. In a clinical context, recall for the Disease class is particularly important — a false negative (missing a sick patient) is more costly than a false positive — and Naive Bayes achieves the best balance across both metrics after tuning.

##Requirements
```
pandas, numpy, matplotlib, scipy, scikit-learn
```
