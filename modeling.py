import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import os
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_classifier(name, y_true, y_pred, report_file):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"{name} → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    with open(report_file, 'a') as f:
        f.write(f"### {name}\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision:** {prec:.4f}\n")
        f.write(f"- **Recall:** {rec:.4f}\n")
        f.write(f"- **F1 Score:** {f1:.4f}\n\n")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def plot_feature_importance(model, feature_names, output_dir, model_name):
    """Plot feature importances for tree-based models."""
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importances[::-1], color='steelblue')
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Features - {model_name}')
        plt.tight_layout()
        path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(path)
        plt.close()
        logger.info(f"Feature importance plot saved to {path}")
    except AttributeError:
        logger.info(f"Feature importance not available for {model_name}")


def run_modeling_pipeline():
    output_dir = "assets"
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "model_comparison_report.md")

    # Check if cleaned data exists
    clean_train_path = "data/cleaned_Weather_Training_Data.csv"
    if not os.path.exists(clean_train_path):
        logger.error(f"Cleaned data {clean_train_path} not found. Please run the data cleaning script first.")
        return

    logger.info("Loading cleaned dataset...")
    df = pd.read_csv(clean_train_path)

    target_col = 'RainTomorrow'
    if target_col not in df.columns:
        target_cols = [c for c in df.columns if 'RainTomorrow' in c]
        if target_cols:
            target_col = target_cols[-1]
        else:
            logger.error("Target column 'RainTomorrow' not found.")
            return

    logger.info(f"Total rows loaded: {len(df)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = list(X.columns)

    # Split data - use more data for training to push accuracy up
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    logger.info(f"Training: {len(X_train)} | Testing: {len(X_test)}")

    # Initialize Report
    with open(report_file, 'w') as f:
        f.write("# Model Comparison Report — Accuracy Target: 92%\n\n")
        f.write(f"**Training samples:** {len(X_train)} | **Test samples:** {len(X_test)}\n\n")

    model_metrics = {}
    trained_models = {}

    # For SVM, subsample to avoid timeout - use larger sample for tree-based models
    SVM_SAMPLE = min(8000, len(X_train))

    # ─────────────────────────────────────────────────────────
    # 1. K-Means Clustering (Elbow Analysis)
    # ─────────────────────────────────────────────────────────
    logger.info("--- 1. K-Means Elbow Analysis ---")
    X_train_svm = X_train.sample(SVM_SAMPLE, random_state=42)
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        km.fit(X_train_svm)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o', color='steelblue')
    plt.title('K-Means Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    elbow_path = os.path.join(output_dir, 'kmeans_elbow_method.png')
    plt.savefig(elbow_path)
    plt.close()

    with open(report_file, 'a') as f:
        f.write("## 1. K-Means Clustering\n")
        f.write(f"Elbow method plot saved to `{elbow_path}`. K=2 aligns with binary target.\n\n")

    # ─────────────────────────────────────────────────────────
    # 2. SVM — GridSearchCV
    # ─────────────────────────────────────────────────────────
    logger.info("--- 2. SVM Hyperparameter Optimization (GridSearchCV) ---")
    X_train_sub = X_train.sample(SVM_SAMPLE, random_state=42)
    y_train_sub = y_train.loc[X_train_sub.index]

    svm_param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'],    'C': [1, 10, 100], 'gamma': ['scale', 0.01]},
    ]
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42, class_weight='balanced'),
        svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    svm_grid.fit(X_train_sub, y_train_sub)
    best_svm = svm_grid.best_estimator_
    logger.info(f"Best SVM Params: {svm_grid.best_params_} | CV Acc: {svm_grid.best_score_:.4f}")

    y_pred_svm = best_svm.predict(X_test)
    metric_svm = evaluate_classifier("Optimized SVM", y_test, y_pred_svm, report_file)
    model_metrics['Optimized SVM'] = metric_svm
    trained_models['Optimized SVM'] = best_svm

    with open(report_file, 'a') as f:
        f.write("## 2. SVM (GridSearchCV)\n")
        f.write(f"- **Best Params:** `{svm_grid.best_params_}`\n")
        f.write(f"- **Best CV Accuracy:** {svm_grid.best_score_:.4f}\n\n")

    # ─────────────────────────────────────────────────────────
    # 3. Decision Tree — GridSearchCV
    # ─────────────────────────────────────────────────────────
    logger.info("--- 3. Decision Tree Hyperparameter Optimization (GridSearchCV) ---")
    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 8, 12, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced'],
    }
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    logger.info(f"Best DT Params: {dt_grid.best_params_} | CV Acc: {dt_grid.best_score_:.4f}")

    y_pred_dt = best_dt.predict(X_test)
    metric_dt = evaluate_classifier("Optimized Decision Tree", y_test, y_pred_dt, report_file)
    model_metrics['Optimized Decision Tree'] = metric_dt
    trained_models['Optimized Decision Tree'] = best_dt

    with open(report_file, 'a') as f:
        f.write("## 3. Decision Tree (GridSearchCV)\n")
        f.write(f"- **Best Params:** `{dt_grid.best_params_}`\n")
        f.write(f"- **Best CV Accuracy:** {dt_grid.best_score_:.4f}\n\n")

    # ─────────────────────────────────────────────────────────
    # 4. Overall Comparison — Select Best Model
    # ─────────────────────────────────────────────────────────
    logger.info("--- 4. Selecting Best Model ---")

    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['accuracy'])
    best_model = trained_models[best_model_name]
    best_acc = model_metrics[best_model_name]['accuracy']

    logger.info(f"🏆 Best Model: {best_model_name} | Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

    if best_acc >= 0.92:
        logger.info("✅ Target of 92% accuracy ACHIEVED!")
    else:
        logger.warning(f"⚠️  Accuracy is {best_acc*100:.2f}%. Consider more data or feature engineering.")

    with open(report_file, 'a') as f:
        f.write("## 7. Overall Leaderboard\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("|---|---|---|---|---|\n")
        for m, met in sorted(model_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            badge = " 🏆" if m == best_model_name else ""
            f.write(f"| {m}{badge} | {met['accuracy']:.4f} | {met['precision']:.4f} | {met['recall']:.4f} | {met['f1']:.4f} |\n")
        f.write(f"\n**Winner:** `{best_model_name}` with **{best_acc*100:.2f}% accuracy**\n\n")
        if best_acc >= 0.92:
            f.write("✅ **Target of 92% accuracy achieved!**\n\n")
        else:
            f.write(f"⚠️ Current best is {best_acc*100:.2f}%. Ensemble or more data may further improve results.\n\n")

    # Save best model
    model_path = os.path.join(output_dir, 'best_model.joblib')
    joblib.dump(best_model, model_path)
    logger.info(f"Best model saved to {model_path}")

    # Save metadata
    meta_path = os.path.join(output_dir, 'best_model_meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"best_model_name={best_model_name}\n")
        f.write(f"accuracy={best_acc:.4f}\n")
        f.write(f"f1_score={model_metrics[best_model_name]['f1']:.4f}\n")

    # Feature importance for tree-based best model
    plot_feature_importance(best_model, feature_names, output_dir, best_model_name)

    # ─────────────────────────────────────────────────────────
    # 8. Confusion Matrix on Test Set
    # ─────────────────────────────────────────────────────────
    logger.info("--- 8. Generating Confusion Matrix ---")
    
    clean_test_path = "data/cleaned_Weather_Test_Data.csv"
    if os.path.exists(clean_test_path):
        df_test = pd.read_csv(clean_test_path)
        has_target = target_col in df_test.columns
        X_final = df_test.copy()
        if has_target:
            X_final = X_final.drop(columns=[target_col])
        X_final = X_final.reindex(columns=feature_names, fill_value=0)

        y_final_pred = best_model.predict(X_final)

        df_test['Predicted_RainTomorrow'] = y_final_pred
        pred_path = os.path.join(output_dir, 'test_predictions.csv')
        df_test.to_csv(pred_path, index=False)

        if has_target:
            y_final = df_test[target_col]
            cm = confusion_matrix(y_final, y_final_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        else:
            cm = confusion_matrix(y_test, best_model.predict(X_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    else:
        # Use validation set
        y_val_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_val_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix_best_model.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    logger.info("========================================================")
    logger.info(f"Pipeline complete. Best model: {best_model_name}")
    logger.info(f"Test Accuracy: {best_acc*100:.2f}%")
    logger.info(f"Results written to: {output_dir}/")
    logger.info("========================================================")


if __name__ == "__main__":
    run_modeling_pipeline()
