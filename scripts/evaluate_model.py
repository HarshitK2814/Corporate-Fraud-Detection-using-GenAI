"""
Model Evaluation Suite (v2 - Conference Ready)
===============================================
Addresses Weaknesses 2, 3, and 6:
  - W2: Evaluation on full 300-company dataset (not 8)
  - W3: Proper ablation study with 6 modality configurations
  - W6: 5-fold CV, SHAP values, bootstrap CIs, paired t-tests

Outputs paper-ready tables and SHAP analysis.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def load_data():
    csv_path = "dataset/real_fraud_benchmark.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run generate_real_dataset.py first.")
    df = pd.read_csv(csv_path)
    y = df["is_fraud"].copy()
    
    # 6 modality configurations for ablation
    modalities = {
        "Geo Only": ["geo_shell_risk"],
        "Audio Only": ["audio_jitter_zscore", "audio_shimmer_zscore", "audio_pitch_variance", "audio_pause_rate"],
        "Text Only": ["text_semantic_evasion"],
        "Audio+Text (Behavioral)": ["audio_jitter_zscore", "audio_shimmer_zscore", "audio_pitch_variance", "audio_pause_rate", "text_semantic_evasion"],
        "Geo+Text": ["geo_shell_risk", "text_semantic_evasion"],
        "CRDI Full Multimodal": ["geo_shell_risk", "audio_jitter_zscore", "audio_shimmer_zscore", "audio_pitch_variance", "audio_pause_rate", "text_semantic_evasion"]
    }
    return df, y, modalities


# ============================================================
# Table 2: Ablation Study
# ============================================================
def evaluate_ablation():
    print("=" * 100)
    print("  TABLE 2: Ablation Study — Modality Contribution Analysis")
    print("  Model: GradientBoostingClassifier | CV: 5-Fold Stratified | Dataset: 276 companies")
    print("=" * 100)
    print(f"\n{'Configuration':<28} | {'Accuracy':>14} | {'Precision':>14} | {'Recall':>14} | {'F1-Score':>14} | {'AUC-ROC':>14}")
    print("-" * 107)
    
    df, y, modalities = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = {}
    all_fold_f1s = {}  # Store per-fold F1 for paired t-test
    
    for mod_name, cols in modalities.items():
        X_mod = df[cols].copy()
        
        # Use GradientBoosting for ALL configs (fair comparison)
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
        )
        
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        
        for train_idx, test_idx in cv.split(X_mod, y):
            X_tr, X_te = X_mod.iloc[train_idx], X_mod.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1]
            
            accs.append(accuracy_score(y_te, preds))
            precs.append(precision_score(y_te, preds, zero_division=0))
            recs.append(recall_score(y_te, preds, zero_division=0))
            f1s.append(f1_score(y_te, preds, zero_division=0))
            aucs.append(roc_auc_score(y_te, probs))
        
        all_results[mod_name] = {
            "accuracy": (np.mean(accs), np.std(accs)),
            "precision": (np.mean(precs), np.std(precs)),
            "recall": (np.mean(recs), np.std(recs)),
            "f1": (np.mean(f1s), np.std(f1s)),
            "auc": (np.mean(aucs), np.std(aucs)),
        }
        all_fold_f1s[mod_name] = f1s
        
        r = all_results[mod_name]
        marker = " **" if mod_name == "CRDI Full Multimodal" else ""
        print(f"  {mod_name:<26} | {r['accuracy'][0]:.4f}+/-{r['accuracy'][1]:.3f} | "
              f"{r['precision'][0]:.4f}+/-{r['precision'][1]:.3f} | "
              f"{r['recall'][0]:.4f}+/-{r['recall'][1]:.3f} | "
              f"{r['f1'][0]:.4f}+/-{r['f1'][1]:.3f} | "
              f"{r['auc'][0]:.4f}+/-{r['auc'][1]:.3f}{marker}")
    
    # Delta analysis
    full_f1 = all_results["CRDI Full Multimodal"]["f1"][0]
    print(f"\n  Multimodal Lift Analysis (F1 vs Full):")
    for mod_name, r in all_results.items():
        if mod_name != "CRDI Full Multimodal":
            delta = full_f1 - r["f1"][0]
            print(f"    CRDI Full vs {mod_name:<26}: +{delta:.4f} F1")
    
    # Save ablation results
    ablation_rows = []
    for name, r in all_results.items():
        ablation_rows.append({
            "Configuration": name,
            "Accuracy": f"{r['accuracy'][0]:.4f} +/- {r['accuracy'][1]:.3f}",
            "F1": f"{r['f1'][0]:.4f} +/- {r['f1'][1]:.3f}",
            "AUC": f"{r['auc'][0]:.4f} +/- {r['auc'][1]:.3f}",
        })
    pd.DataFrame(ablation_rows).to_csv("dataset/ablation_results.csv", index=False)
    print("\n  Ablation results saved to dataset/ablation_results.csv")
    
    return all_results, all_fold_f1s, df, y


# ============================================================
# Statistical Significance Testing
# ============================================================
def test_statistical_significance(df, y, all_fold_f1s):
    print("\n" + "=" * 100)
    print("  TABLE 3: Statistical Significance Analysis")
    print("=" * 100)
    
    full_cols = ["geo_shell_risk", "audio_jitter_zscore", "audio_shimmer_zscore",
                 "audio_pitch_variance", "audio_pause_rate", "text_semantic_evasion"]
    
    # --- Part A: Paired t-test (fold-level F1) ---
    print("\n  A. Paired t-test (per-fold F1: CRDI Full vs each baseline):")
    print(f"  {'Comparison':<45} | {'t-stat':>8} | {'p-value':>8} | {'Significant':>12}")
    print("  " + "-" * 82)
    
    full_f1s = all_fold_f1s["CRDI Full Multimodal"]
    
    for mod_name, fold_f1s in all_fold_f1s.items():
        if mod_name == "CRDI Full Multimodal":
            continue
        t_stat, p_val = stats.ttest_rel(full_f1s, fold_f1s)
        sig = "YES (p<0.05)" if p_val < 0.05 else "NO"
        print(f"  Full vs {mod_name:<35} | {t_stat:>8.4f} | {p_val:>8.4f} | {sig:>12}")
    
    # --- Part B: Bootstrap Confidence Intervals ---
    print(f"\n  B. Bootstrap 95% CI for F1 Improvement (Full vs Text Only):")
    
    n_bootstrap = 1000
    n_size = int(len(df) * 0.8)
    
    text_cols = ["text_semantic_evasion"]
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42)
    
    diff_f1s = []
    for i in range(n_bootstrap):
        X_boot, y_boot = resample(df, y, n_samples=n_size, random_state=i, stratify=y)
        
        split_pt = int(n_size * 0.8)
        X_train, X_test = X_boot.iloc[:split_pt], X_boot.iloc[split_pt:]
        y_train, y_test = y_boot.iloc[:split_pt], y_boot.iloc[split_pt:]
        
        # Full model
        model.fit(X_train[full_cols], y_train)
        f1_full = f1_score(y_test, model.predict(X_test[full_cols]), zero_division=0)
        
        # Text baseline
        model.fit(X_train[text_cols], y_train)
        f1_text = f1_score(y_test, model.predict(X_test[text_cols]), zero_division=0)
        
        diff_f1s.append(f1_full - f1_text)
    
    diff_mean = np.mean(diff_f1s)
    ci_lower = np.percentile(diff_f1s, 2.5)
    ci_upper = np.percentile(diff_f1s, 97.5)
    p_value = sum(np.array(diff_f1s) <= 0) / float(n_bootstrap)
    
    print(f"     Mean F1 Improvement:  +{diff_mean:.4f}")
    print(f"     95% Confidence Int:   [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"     Bootstrap p-value:    {p_value:.4f}")
    sig_text = "STATISTICALLY SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
    print(f"     Verdict:              {sig_text}")


# ============================================================
# SHAP Explainability
# ============================================================
def generate_shap_analysis(df, y):
    print("\n" + "=" * 100)
    print("  TABLE 4: SHAP Feature Importance (Global + Local)")
    print("=" * 100)
    
    try:
        import shap
    except ImportError:
        print("  [!] SHAP not installed. Run: pip install shap")
        print("  [!] Falling back to permutation importance.\n")
        generate_permutation_importance(df, y)
        return
    
    feature_cols = ["geo_shell_risk", "audio_jitter_zscore", "audio_shimmer_zscore",
                    "audio_pitch_variance", "audio_pause_rate", "text_semantic_evasion"]
    
    X = df[feature_cols].copy()
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
    )
    model.fit(X, y)
    
    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Global importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total = mean_abs_shap.sum()
    
    print(f"\n  A. Global SHAP Feature Importance (mean |SHAP value|):")
    print(f"  {'Feature':<28} | {'Mean |SHAP|':>12} | {'% Contribution':>15}")
    print("  " + "-" * 62)
    
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    for idx in sorted_idx:
        pct = mean_abs_shap[idx] / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {feature_cols[idx]:<28} | {mean_abs_shap[idx]:>12.4f} | {pct:>13.1f}%  {bar}")
    
    # Local SHAP for 3 interesting cases
    print(f"\n  B. Local SHAP Explanations (3 sample companies):")
    
    fraud_idx = df[df["is_fraud"] == 1].index[:1].tolist()
    normal_idx = df[df["is_fraud"] == 0].index[:1].tolist()
    # Find a borderline case (model probability closest to 0.5)
    probs = model.predict_proba(X)[:, 1]
    borderline_idx = [np.argmin(np.abs(probs - 0.5))]
    
    for case_name, idx_list in [("Known Fraud", fraud_idx), ("Known Normal", normal_idx), ("Borderline", borderline_idx)]:
        if not idx_list:
            continue
        idx = idx_list[0]
        company = df.iloc[idx]["company_name"]
        prob = probs[idx]
        print(f"\n    {case_name}: {company} (P(fraud)={prob:.4f})")
        sv = shap_values[idx]
        for i in sorted_idx:
            direction = "+" if sv[i] > 0 else "-"
            print(f"      {feature_cols[i]:<25}: SHAP={sv[i]:>+.4f}  (value={X.iloc[idx, i]:.4f})")
    
    # Save SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    shap_df["company_name"] = df["company_name"].values
    shap_df["is_fraud"] = y.values
    shap_df.to_csv("dataset/shap_values.csv", index=False)
    print(f"\n  SHAP values saved to dataset/shap_values.csv")


def generate_permutation_importance(df, y):
    """Fallback if SHAP is not installed."""
    from sklearn.inspection import permutation_importance
    
    feature_cols = ["geo_shell_risk", "audio_jitter_zscore", "audio_shimmer_zscore",
                    "audio_pitch_variance", "audio_pause_rate", "text_semantic_evasion"]
    X = df[feature_cols].copy()
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
    )
    model.fit(X, y)
    
    result = permutation_importance(model, X, y, n_repeats=30, random_state=42, scoring='f1')
    
    print(f"\n  Permutation Feature Importance (30 repeats):")
    print(f"  {'Feature':<28} | {'Importance':>12} | {'Std':>8}")
    print("  " + "-" * 55)
    
    sorted_idx = result.importances_mean.argsort()[::-1]
    for idx in sorted_idx:
        print(f"  {feature_cols[idx]:<28} | {result.importances_mean[idx]:>12.4f} | {result.importances_std[idx]:>8.4f}")


# ============================================================
# Dataset Distribution Summary
# ============================================================
def print_dataset_summary(df, y):
    print("\n" + "=" * 100)
    print("  TABLE 1: Dataset Characteristics")
    print("=" * 100)
    
    feature_cols = ["geo_shell_risk", "audio_jitter_zscore", "audio_shimmer_zscore",
                    "audio_pitch_variance", "audio_pause_rate", "text_semantic_evasion"]
    
    print(f"\n  Total samples: {len(df)} | Fraud: {y.sum()} ({y.sum()/len(y)*100:.1f}%) | Normal: {len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    if "size_category" in df.columns:
        print(f"  Size distribution: {dict(df['size_category'].value_counts())}")
    
    print(f"\n  {'Feature':<28} | {'Fraud (mean+/-std)':>20} | {'Normal (mean+/-std)':>20} | {'Cohen d':>10}")
    print("  " + "-" * 86)
    
    for col in feature_cols:
        fraud_vals = df[df["is_fraud"]==1][col]
        normal_vals = df[df["is_fraud"]==0][col]
        
        # Cohen's d for effect size
        pooled_std = np.sqrt((fraud_vals.std()**2 + normal_vals.std()**2) / 2)
        cohens_d = (fraud_vals.mean() - normal_vals.mean()) / pooled_std if pooled_std > 0 else 0
        
        print(f"  {col:<28} | {fraud_vals.mean():>7.3f} +/- {fraud_vals.std():>5.3f}  | "
              f"{normal_vals.mean():>7.3f} +/- {normal_vals.std():>5.3f}  | {cohens_d:>8.3f}")
    
    print("\n  Cohen's d interpretation: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Load
    df, y, modalities = load_data()
    
    # Table 1: Dataset summary
    print_dataset_summary(df, y)
    
    # Table 2: Ablation study
    all_results, all_fold_f1s, df, y = evaluate_ablation()
    
    # Table 3: Statistical significance
    test_statistical_significance(df, y, all_fold_f1s)
    
    # Table 4: SHAP analysis
    generate_shap_analysis(df, y)
    
    print("\n" + "=" * 100)
    print("  Evaluation Suite Complete. All tables ready for paper insertion.")
    print("=" * 100)
