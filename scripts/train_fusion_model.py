"""
Train ML Fusion Model (v2 - Conference Ready)
==============================================
Addresses Weakness 6: Proper 5-fold stratified CV with mean±std.
Uses the same GradientBoosting model with regularization.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, make_scorer)


def load_data(csv_path="dataset/real_fraud_benchmark.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run generate_real_dataset.py first.")
    
    df = pd.read_csv(csv_path)
    feature_cols = [
        "geo_shell_risk",
        "audio_jitter_zscore",
        "audio_shimmer_zscore",
        "audio_pitch_variance",
        "audio_pause_rate",
        "text_semantic_evasion"
    ]
    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()
    
    return X, y, feature_cols, df


def train_and_compare():
    print("=" * 85)
    print("  CRDI Fusion Model Training (v2 - 5-Fold Stratified CV)")
    print("=" * 85)
    
    X, y, feature_cols, df = load_data()
    
    print(f"\n  Dataset: {len(df)} companies | Fraud: {y.sum()} | Normal: {len(y) - y.sum()}")
    print(f"  Features: {feature_cols}\n")
    
    # Define all models with regularization (anti-overfitting)
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.05, class_weight='balanced', max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, max_depth=3,
            class_weight='balanced', random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(16, 8), max_iter=1000, alpha=0.1, random_state=42
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    # Custom scoring
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'auc': 'roc_auc'
    }
    
    # Table header
    print(f"{'Model':<25} | {'Accuracy':>12} | {'Precision':>12} | {'Recall':>12} | {'F1-Score':>12} | {'AUC-ROC':>12}")
    print("-" * 100)
    
    best_f1 = 0
    best_model_name = ""
    best_model = None
    best_is_tree = False
    all_results = {}
    
    for name, model in models.items():
        is_tree = name in ["Random Forest", "Gradient Boosting"]
        
        # Manual CV to handle scaling properly
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            if not is_tree:
                sc = StandardScaler()
                X_tr_use = pd.DataFrame(sc.fit_transform(X_tr), columns=feature_cols)
                X_te_use = pd.DataFrame(sc.transform(X_te), columns=feature_cols)
            else:
                X_tr_use, X_te_use = X_tr, X_te
            
            model.fit(X_tr_use, y_tr)
            y_pred = model.predict(X_te_use)
            y_prob = model.predict_proba(X_te_use)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
            
            accs.append(accuracy_score(y_te, y_pred))
            precs.append(precision_score(y_te, y_pred, zero_division=0))
            recs.append(recall_score(y_te, y_pred, zero_division=0))
            f1s.append(f1_score(y_te, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_te, y_prob))
        
        mean_f1 = np.mean(f1s)
        all_results[name] = {
            "accuracy": (np.mean(accs), np.std(accs)),
            "precision": (np.mean(precs), np.std(precs)),
            "recall": (np.mean(recs), np.std(recs)),
            "f1": (np.mean(f1s), np.std(f1s)),
            "auc": (np.mean(aucs), np.std(aucs)),
        }
        
        r = all_results[name]
        print(f"{name:<25} | {r['accuracy'][0]:.4f}+/-{r['accuracy'][1]:.3f} | "
              f"{r['precision'][0]:.4f}+/-{r['precision'][1]:.3f} | "
              f"{r['recall'][0]:.4f}+/-{r['recall'][1]:.3f} | "
              f"{r['f1'][0]:.4f}+/-{r['f1'][1]:.3f} | "
              f"{r['auc'][0]:.4f}+/-{r['auc'][1]:.3f}")
        
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = name
            best_model = model
            best_is_tree = is_tree
    
    print(f"\n  Best Model: {best_model_name} (F1: {best_f1:.4f})")
    
    # Retrain best model on full data for deployment
    if not best_is_tree:
        final_scaler = StandardScaler()
        X_final = pd.DataFrame(final_scaler.fit_transform(X), columns=feature_cols)
    else:
        final_scaler = None
        X_final = X
    
    best_model.fit(X_final, y)
    
    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        print(f"\n  Feature Importances ({best_model_name}):")
        importances = best_model.feature_importances_
        for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
            bar = "#" * int(imp * 50)
            print(f"    {feat:<25}: {imp:.4f}  {bar}")
    elif hasattr(best_model, "coef_"):
        print(f"\n  Coefficient Magnitudes ({best_model_name}):")
        coefs = np.abs(best_model.coef_[0])
        for feat, c in sorted(zip(feature_cols, coefs), key=lambda x: x[1], reverse=True):
            bar = "#" * int(c * 10)
            print(f"    {feat:<25}: {c:.4f}  {bar}")
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    export_dict = {
        "model": best_model,
        "scaler": final_scaler,
        "is_tree": best_is_tree,
        "features": feature_cols,
        "cv_results": all_results,
        "dataset_size": len(df),
        "n_folds": 5
    }
    joblib.dump(export_dict, f"{model_dir}/crdi_fusion_model.pkl")
    print(f"\n  Model exported to {model_dir}/crdi_fusion_model.pkl")
    
    # Save results CSV for paper
    results_rows = []
    for name, r in all_results.items():
        results_rows.append({
            "Model": name,
            "Accuracy_Mean": r["accuracy"][0], "Accuracy_Std": r["accuracy"][1],
            "Precision_Mean": r["precision"][0], "Precision_Std": r["precision"][1],
            "Recall_Mean": r["recall"][0], "Recall_Std": r["recall"][1],
            "F1_Mean": r["f1"][0], "F1_Std": r["f1"][1],
            "AUC_Mean": r["auc"][0], "AUC_Std": r["auc"][1],
        })
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("dataset/model_comparison_results.csv", index=False)
    print("  Results saved to dataset/model_comparison_results.csv\n")


if __name__ == "__main__":
    train_and_compare()
