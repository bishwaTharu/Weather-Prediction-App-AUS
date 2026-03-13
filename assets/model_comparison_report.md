# Model Comparison Report — Accuracy Target: 92%

**Training samples:** 131166 | **Test samples:** 23148

## 1. K-Means Clustering
Elbow method plot saved to `assets/kmeans_elbow_method.png`. K=2 aligns with binary target.

### Optimized SVM
- **Accuracy:** 0.7951
- **Precision:** 0.8030
- **Recall:** 0.7820
- **F1 Score:** 0.7923

## 2. SVM (GridSearchCV)
- **Best Params:** `{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}`
- **Best CV Accuracy:** 0.7948

### Optimized Decision Tree
- **Accuracy:** 0.9172
- **Precision:** 0.8715
- **Recall:** 0.9787
- **F1 Score:** 0.9220

## 3. Decision Tree (GridSearchCV)
- **Best Params:** `{'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}`
- **Best CV Accuracy:** 0.8810

## 7. Overall Leaderboard

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Optimized Decision Tree 🏆 | 0.9172 | 0.8715 | 0.9787 | 0.9220 |
| Optimized SVM | 0.7951 | 0.8030 | 0.7820 | 0.7923 |

**Winner:** `Optimized Decision Tree` with **91.72% accuracy**

⚠️ Current best is 91.72%. Ensemble or more data may further improve results.

