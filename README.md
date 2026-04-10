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

##Project 4 — TMDB Movies: Unsupervised Learning

Notebook: tmdb_unsupervised.ipynb
Dataset: tmdb_5000_movies.csv — 4,803 movies, 20 features including budget, revenue, genres, keywords, and ratings
What the notebook covers

##Feature engineering from JSON columns (genres, keywords, production companies, countries, languages)

Handling zero values in budget and revenue (treated as missing)
StandardScaler preprocessing
K-Means clustering with Elbow method (SSE) and Silhouette score for optimal k selection
DBSCAN clustering with grid search over eps and min_samples
Silhouette evaluation and per-sample silhouette plot
PCA 2D projection for cluster visualisation
Sample movies per cluster and cluster profile analysis
Side-by-side comparison of K-Means vs DBSCAN

##Engineered features

FeatureDescriptionbudgetProduction budget (zeros treated as missing)revenueBox office revenue (zeros treated as missing)runtimeFilm duration in minutespopularityTMDB popularity scorevote_averageAverage user ratingvote_countNumber of user votesnum_genresNumber of genres listednum_keywordsNumber of keywords listednum_production_companiesNumber of production companiesnum_countriesNumber of production countriesnum_languagesNumber of spoken languagesis_english1 if original language is English
Key findings
After dropping rows with missing budget or revenue, 3,229 movies remained for clustering. K-Means with k=3 achieved the best Silhouette score (0.37), separating movies into low-budget/low-revenue independent films, mid-range productions, and large-scale blockbusters. DBSCAN with eps=0.5 and min_samples=5 achieved a stronger Silhouette of 0.59 on non-noise points, identifying tighter and more compact clusters while flagging outliers — films with unusual combinations of scale, popularity, and genre breadth.
Observations

Over 1,000 movies have a budget of zero and over 1,400 have a revenue of zero in the raw dataset, which are missing values rather than true zeros. Treating these as missing and dropping affected rows reduces the dataset to 3,229 movies but ensures that clustering is based on real financial data rather than artefacts of incomplete records.
The Elbow method shows a gradual decline in SSE without a sharp bend, which is typical for real-world movie data where clusters are not perfectly separated. The Silhouette score peaks at k=3, suggesting three broad groupings are the most natural partition of this dataset.
K-Means with k=3 separates movies primarily along production scale — one cluster captures small independent productions with low budgets, low revenue, and low vote counts; a second captures mid-range studio films; and a third captures major blockbusters with high budgets, high revenue, high popularity, and large vote counts. The genre and keyword counts also increase progressively across clusters, reflecting the richer metadata coverage of bigger productions.
DBSCAN achieves a higher Silhouette score than K-Means on the clustered points (0.59 vs 0.37), indicating that the dense regions it identifies are more internally coherent. It also flags outlier movies that do not conform to any cluster — typically films with extreme popularity or unusual financial profiles relative to their genre and production scale, such as very low-budget films that became unexpectedly popular.
PCA reveals that the first two principal components capture a meaningful portion of variance, with PC1 primarily driven by financial scale (budget, revenue, vote count) and PC2 capturing genre and language diversity. This confirms that production scale is the dominant organising principle in this dataset, and that unsupervised learning can recover a commercially meaningful segmentation of the film industry without any labelled data.


##Requirements
```
pandas, numpy, matplotlib, scipy, scikit-learn
```

##Project 5 — Titanic Survival Prediction: Neural Networks
Notebook: titanic_nn.ipynb
Dataset: Titanic-Dataset.csv — 891 passengers, survival classification
What the notebook covers

##Data preprocessing: dropping irrelevant columns, median/mode imputation, label encoding, one-hot encoding
Stratified train/test split (80/20) and StandardScaler normalisation
Model 1: Basic Neural Network — 1 hidden layer (16 units, ReLU, sigmoid output)
Model 2: Deep Neural Network — 3 hidden layers (128 → 64 → 32 units) with L2 regularisation and Dropout
Training history plots: loss and accuracy over epochs for both models
Confusion matrices and classification reports
ROC curve and AUC comparison between both models
Hyperparameter experiments across 4 configurations (learning rate, batch size, L2, dropout)

##Model architectures
ModelLayersRegularisationEpochsEarly StoppingBasic NNInput → 16 → 1None80NoDeep NNInput → 128 → 64 → 32 → 1L2 + Dropout (0.3)up to 150Yes (patience=10)
Hyperparameter configurations tested
LRBatchL2Dropout0.01160.0010.20.001320.010.30.0001640.0010.40.005320.050.3
Key findings
Both models achieve competitive accuracy on the Titanic dataset. The Deep NN benefits from L2 regularisation and Dropout to prevent overfitting on a small dataset of 891 samples. Early stopping with patience=10 prevents unnecessary training. Hyperparameter experiments show that lr=0.001 with batch=32 provides the best balance between convergence speed and generalisation. AUC is used alongside accuracy to account for the slight class imbalance between survivors and non-survivors.
Observations

The Titanic dataset is small (891 rows), which makes overfitting a real concern for neural networks. The basic model with a single hidden layer of 16 units is deliberately kept shallow to serve as a stable baseline, while the deep model compensates for its greater capacity with L2 regularisation and Dropout at each major hidden layer.
Early stopping on validation loss is essential for the Deep NN. Without it, the model continues to minimise training loss while validation loss starts to rise — a textbook overfitting pattern. With patience=10 and restore_best_weights=True, the model automatically reverts to the checkpoint with the best generalisation performance.
A learning rate of 0.001 with the Adam optimiser consistently outperforms both higher and lower alternatives in the hyperparameter experiments. Higher learning rates (0.01, 0.005) cause unstable validation loss curves, while a very low rate (0.0001) converges too slowly and often stops before reaching a good solution within the epoch budget.
AUC is a more informative evaluation metric than accuracy alone for this dataset. The Titanic survival rate is approximately 38%, creating a moderate class imbalance where a naive classifier predicting "not survived" would still achieve ~62% accuracy. The ROC curve and AUC measure the model's ability to rank survivors above non-survivors regardless of the classification threshold, providing a clearer picture of discriminative power.
On a tabular dataset of this size, the performance gap between the Basic NN and the Deep NN is smaller than one might expect. The additional layers in the Deep NN do not automatically guarantee better results — regularisation choices, learning rate, and early stopping matter more than raw model depth. This reflects a broader principle in applied machine learning: on small structured datasets, model complexity should be scaled carefully to avoid the regularisation burden outweighing the capacity benefit.

##Requirements
```
pandas, numpy, matplotlib, scipy, scikit-learn
```
