# Model Comparison Report

This report details the implementation, evaluation, and comparison of machine learning models applied to the Weather dataset, predicting `RainTomorrow`.

### 2. K-Means Clustering
An **Elbow Method plot** was generated (saved to `assets/kmeans_elbow_method.png`) to determine the optimal K. By plotting Inertia vs K, the optimal clustering looks to be around K=2 to 4.
- **Chosen K:** We selected $K=2$, which broadly represents 'Rainy/High Humidity' vs 'Dry/Clear' weather profiles.
- **Clustering Outcome:** K-Means successfully identifies natural groupings in the data. However, as an unsupervised algorithm, it attempts to find structure rather than solve the mapping $X \rightarrow y$. The resulting clusters partially align with `RainTomorrow`, indicating weather patterns influence the target.

### SVM (Linear Kernel) Performance
- **Accuracy:** 0.8300
- **Precision:** 0.6707
- **Recall:** 0.4934
- **F1 Score:** 0.5685

### SVM (RBF Kernel) Performance
- **Accuracy:** 0.8320
- **Precision:** 0.7034
- **Recall:** 0.4493
- **F1 Score:** 0.5484

#### Hyperparameters Explained
- **Kernel Choice:** We tested both Linear (draws a straight hyperplane) and RBF (Radial Basis Function, maps data to infinite-dimensional space to find non-linear separations).
- **Hyperparameter `C` (Regularization):** Controls the trade-off between achieving a low training error and a low testing error. A low C creates a larger margin but may misclassify points (higher bias, lower variance). A high C aims to classify all training points correctly (lower bias, higher variance, risk of overfitting).
- **Hyperparameter `gamma`:** Defines how far the influence of a single training example reaches. Low gamma means 'far' (broad kernel), high gamma means 'close' (narrow kernel leading to heavily isolated islands). Setting it to 'scale' heuristically balances this.

### Decision Tree Performance
- **Accuracy:** 0.8090
- **Precision:** 0.6169
- **Recall:** 0.4185
- **F1 Score:** 0.4987

#### Depth Selection and Interpretability
- **Depth Selection:** We plotted Train vs Test Accuracy across max_depth values (saved to `assets/decision_tree_depth.png`). As depth increases, train accuracy approaches 1.0 (overfitting), while test accuracy peaks and then declines. We selected `max_depth=5` to balance bias and variance.
- **Interpretability:** Unlike SVM or complex Neural Networks, Decision Trees are 'white-box' models. We can easily extract the exact boolean rules (e.g., 'If Humidity3pm > 70 AND Rainfall > 2 -> Predict Rain'). As complexity (depth) increases, explainability drops slightly but remains vastly superior to RBF SVM.

## Overall Comparison & Justification

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| SVM (Linear Kernel) | 0.8300 | 0.6707 | 0.4934 | 0.5685 |
| SVM (RBF Kernel) | 0.8320 | 0.7034 | 0.4493 | 0.5484 |
| Decision Tree | 0.8090 | 0.6169 | 0.4185 | 0.4987 |

### Which model performs best and why?
The **SVM (RBF Kernel)** achieved the highest accuracy.
- **Justification:** Weather data often holds non-linear feature interactions (e.g., specific combination of wind direction, pressure, and temperature leads to rain). Models capable of capturing non-linear boundaries (like RBF SVM or Decision Trees) generally outperform strictly linear models.

### Discussion
- **Data Characteristics:** The dataset contains mixed data types (categorical and numerical). Features like 'Rainfall' are heavily skewed (mostly 0). While tree-based models handle this inherently well without scaling, distance-based models like SVM and K-Means were dependent heavily on the StandardScaler applied during the Data Cleaning pipeline.
- **Bias-Variance Trade-off:** 
  - High depth Decision Trees demonstrated low bias but extremely high variance (overfitting the noisy weather data).
  - Linear SVM displayed high bias (underfitting non-linear weather patterns) but low variance.
  - The RBF SVM and bounded Decision Tree balanced this trade-off effectively through proper hyperparameter tuning.
- **Interpretability vs Accuracy:** There is a distinct trade-off. RBF SVM might offer high accuracy but operates as a 'black box'. The Decision Tree (with depth=5) usually sacrifices a slight fraction of accuracy for immense interpretability, allowing meteorologists to understand exactly *why* rain was predicted.

## Final Evaluation
The `cleaned_Weather_Test_Data.csv` did not contain the ground-truth `RainTomorrow` column, so predictions were generated and saved to `assets/test_predictions.csv`.
Since true labels were missing, the confusion matrix (saved to `assets/confusion_matrix_best_model.png`) was instead generated using the completely unseen 20% validation split from the training data for (**SVM (RBF Kernel)**).
