"""
Fusion Scorer   CRDI  Truth Score Generator

Combines geospatial verification and behavioural forensics
into a single Corporate Reality Distortion Index (CRDI) score.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class FusionScorer:
    """
    Merges two analysis streams:
      1. Geospatial shell-risk score (0 1)
      2. Behavioural/voice combined score (0 1)

    Produces a final CRDI Truth Score using either an advanced ML Fusion Model
    or heuristic baseline weights.
    """

    # Default weights (baseline heuristic if ML fails or missing data)
    DEFAULT_WEIGHTS = {
        "geospatial": 0.40,
        "behavioral": 0.60,
    }

    RISK_BANDS = [
        (0.30, "CREDIBLE",     " ", "Low distortion risk   indicators appear normal"),
        (0.50, "WATCHLIST",    " ", "Minor anomalies detected   warrants monitoring"),
        (0.70, "SUSPICIOUS",   " ", "Multiple distortion signals   heightened scrutiny recommended"),
        (0.85, "HIGH_RISK",    " ", "Significant distortion indicators   formal investigation warranted"),
        (1.01, "CRITICAL",     " ", "Extreme distortion detected   immediate review required"),
    ]

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.ml_model = None
        self.ml_scaler = None
        self.ml_is_tree = False
        
        # Try to load the trained ML fusion model
        model_path = "models/crdi_fusion_model.pkl"
        if os.path.exists(model_path):
            try:
                ml_dict = joblib.load(model_path)
                self.ml_model = ml_dict.get("model")
                self.ml_scaler = ml_dict.get("scaler")
                self.ml_is_tree = ml_dict.get("is_tree", False)
                print(f"[+] FusionScorer loaded advanced ML model: {type(self.ml_model).__name__}")
            except Exception as e:
                print(f"[-] Failed to load ML fusion model, falling back to heuristics: {e}")

    def generate_score(
        self,
        company_name: str,
        geospatial_result: Optional[Dict[str, Any]] = None,
        behavioral_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate the final CRDI score using ML (if available) or heuristics.
        """
        geo_available = geospatial_result is not None
        beh_available = behavioral_result is not None

        # Extract features safely
        geo_score = geospatial_result.get("avg_shell_risk", 0.5) if geo_available else 0.5
        beh_score = behavioral_result.get("combined_behavioral_score", 0.5) if beh_available else 0.5
        
        sem = behavioral_result.get("semantic_analysis", {}) if beh_available else {}
        audio = behavioral_result.get("audio_analysis", {}) if beh_available else {}

        # Default heuristic score calculation
        if geo_available and not beh_available:
            w_geo, w_beh = 1.0, 0.0
        elif beh_available and not geo_available:
            w_geo, w_beh = 0.0, 1.0
        else:
            w_geo = self.weights["geospatial"]
            w_beh = self.weights["behavioral"]

        heuristic_score = geo_score * w_geo + beh_score * w_beh
        crdi_score = heuristic_score # Default to heuristic
        used_ml = False
        feature_contributions = {}  # Per-company local contributions

        import pandas as pd
        feature_names = [
            "geo_shell_risk", 
            "audio_jitter_zscore", 
            "audio_shimmer_zscore", 
            "audio_pitch_variance", 
            "audio_pause_rate", 
            "text_semantic_evasion"
        ]

        # Apply Advanced ML Fusion Prediction if fully available
        if self.ml_model and geo_available and beh_available:
            try:
                # Construct feature vector [geo, jitter, shimmer, pitch, pause, semantic]
                features = [
                    geo_score,
                    audio.get("jitter", 0.0),
                    audio.get("shimmer", 0.0),
                    audio.get("pitch_std", 0.0),
                    audio.get("pause_rate", 0.0),
                    sem.get("risk_score", 0.0)
                ]
                sem_evasion = sem.get("deception_score", sem.get("risk_score", 0.5))
                features[5] = sem_evasion
                
                X = pd.DataFrame([features], columns=feature_names)
                
                # Scale if necessary
                if not self.ml_is_tree and self.ml_scaler:
                    X = self.ml_scaler.transform(X)
                
                # Predict Probability (Fraud Class = 1)
                if hasattr(self.ml_model, "predict_proba"):
                    ml_score = self.ml_model.predict_proba(X)[0][1]
                else:
                    ml_score = float(self.ml_model.predict(X)[0])
                    
                crdi_score = ml_score
                used_ml = True

                # --- Per-Company Local Feature Contributions ---
                if hasattr(self.ml_model, "feature_importances_"):
                    # For tree models: weight = global importance * this company's feature value
                    imps = self.ml_model.feature_importances_
                    raw_vals = np.array(features)
                    local_weights = imps * np.abs(raw_vals)
                    total_local = local_weights.sum() if local_weights.sum() > 0 else 1.0
                    for i, fn in enumerate(feature_names):
                        feature_contributions[fn] = {
                            "global_importance": round(float(imps[i]), 4),
                            "this_company_value": round(float(features[i]), 4),
                            "local_contribution_pct": round(float(local_weights[i] / total_local * 100), 2)
                        }
                elif hasattr(self.ml_model, "coef_"):
                    coefs = self.ml_model.coef_[0]
                    raw_vals = np.array(features)
                    local_weights = np.abs(coefs * raw_vals)
                    total_local = local_weights.sum() if local_weights.sum() > 0 else 1.0
                    for i, fn in enumerate(feature_names):
                        feature_contributions[fn] = {
                            "coefficient": round(float(coefs[i]), 4),
                            "this_company_value": round(float(features[i]), 4),
                            "local_contribution_pct": round(float(local_weights[i] / total_local * 100), 2)
                        }
            except Exception as e:
                print(f"[!] ML Fusion Inference Error: {e}")

        # Determine risk band
        risk_label, icon, description = "UNKNOWN", " ", ""
        for threshold, label, ico, desc in self.RISK_BANDS:
            if crdi_score < threshold:
                risk_label, icon, description = label, ico, desc
                break

        # Build breakdown
        breakdown = []
        # Determine dynamic percentage contribution for Explainability
        if used_ml and hasattr(self.ml_model, "coef_"):
            # Logistic Regression / Linear Model Explainability
            coefs = np.abs(self.ml_model.coef_[0])
            w_geo_dyn = coefs[0]
            w_beh_dyn = np.sum(coefs[1:])
            total = w_geo_dyn + w_beh_dyn
            weight_geo_str = f"{w_geo_dyn / total:.0%} (ML Learned)"
            weight_beh_str = f"{w_beh_dyn / total:.0%} (ML Learned)"
        elif used_ml and hasattr(self.ml_model, "feature_importances_"):
            # Tree-based Explainability
            imps = self.ml_model.feature_importances_
            w_geo_dyn = imps[0]
            w_beh_dyn = np.sum(imps[1:])
            total = w_geo_dyn + w_beh_dyn
            weight_geo_str = f"{w_geo_dyn / total:.0%} (ML Learned)"
            weight_beh_str = f"{w_beh_dyn / total:.0%} (ML Learned)"
        else:
            weight_geo_str = "ML Dynamic" if used_ml else w_geo
            weight_beh_str = "ML Dynamic" if used_ml else w_beh

        if geo_available:
            breakdown.append({
                "module": "Geospatial Verification",
                "score": round(geo_score, 4),
                "weight": weight_geo_str,
                "verdict": geospatial_result.get("verdict", "N/A"),
                "details": {
                    "images_analyzed": geospatial_result.get("num_images", 0),
                    "max_risk": geospatial_result.get("max_shell_risk", 0),
                },
            })
        if beh_available:
            sem = behavioral_result.get("semantic_analysis", {})
            beh_verdict = behavioral_result.get("verdict")
            if not beh_verdict:
                # If no explicit verdict, generate one based on behavioral score directly
                if beh_score < 0.3: beh_verdict = "Low Risk (Normal Speech)"
                elif beh_score < 0.7: beh_verdict = "Medium Risk (Some Anomalies)"
                else: beh_verdict = "High Risk (Evasive/Stress Detected)"

            breakdown.append({
                "module": "Behavioral Forensics",
                "score": round(beh_score, 4),
                "weight": weight_beh_str,
                "verdict": beh_verdict,
                "details": {
                    "audio_stress": behavioral_result.get("audio_analysis", {}).get("stress_score", 0),
                    "semantic_deception": sem.get("deception_score", 0),
                    "hedging_trend": sem.get("hedging_trend", "N/A"),
                    "evasion_trend": sem.get("evasion_trend", "N/A"),
                    "key_red_flags": sem.get("key_red_flags", [])[:5],
                },
            })

        # Generate dynamic LLM explanations
        llm_rationale = self._generate_llm_rationale(
            company_name, 
            round(crdi_score * 100, 2), 
            geo_score, 
            beh_score, 
            audio.get("stress_score", audio.get("jitter", 0.0)), 
            sem.get("deception_score", sem.get("risk_score", 0.0))
        )
        weight_rationale = self._generate_weight_rationale(
            company_name,
            round(crdi_score * 100, 2),
            risk_label,
            feature_contributions,
            type(self.ml_model).__name__ if self.ml_model else "Heuristic"
        )

        return {
            "company": company_name,
            "crdi_score": round(crdi_score, 4),
            "risk_label": risk_label,
            "risk_icon": icon,
            "risk_description": description,
            "breakdown": breakdown,
            "feature_contributions": feature_contributions,
            "dynamic_rationale": llm_rationale,
            "weight_rationale": weight_rationale,
            "data_sources": {
                "geospatial": geo_available,
                "behavioral": beh_available,
                "used_ml_fusion": used_ml
            },
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_llm_rationale(self, company, crdi_score, geo_score, beh_score, audio_sec, text_sec):
        """Uses Groq Llama3.3 to dynamically write a 2-sentence justification for the score."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
             return "Dynamic explanation unavailable (API key missing)."
             
        try:
             client = Groq(api_key=api_key)
             prompt = f"""
             You are a forensic evaluation AI. Explicitly justify why the company '{company}' received a fraud probability score of {crdi_score}/100 based ON THESE EXACT PIPELINE METRICS:
             - Geospatial Shell Risk: {geo_score}
             - Overall Behavioral Risk: {beh_score}
               (Audio Stress Component: {audio_sec})
               (Linguistics Deception Component: {text_sec})
               
             Keep it to exactly 2 sentences. Be factual, professional, and explain the main reason for the score (e.g., "The critical score is driven by extreme audio stress and a failing geospatial scan..." OR "The credible score is due to passing the corporate map verification combined with calm audio...").
             """
             chat = client.chat.completions.create(
                 messages=[{"role": "user", "content": prompt}],
                 model="llama-3.3-70b-versatile",
                 temperature=0.0
             )
             return chat.choices[0].message.content.strip()
        except Exception as e:
             return f"Failed to generate dynamic rationale: {e}"

    def _generate_weight_rationale(self, company, crdi_score, risk_label, feature_contributions, model_name):
        """Uses Groq Llama3.3 to dynamically explain the per-company weight distribution."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Weight explanation unavailable (API key missing)."
        if not feature_contributions:
            return "Weight explanation unavailable (no feature contributions computed)."

        try:
            client = Groq(api_key=api_key)

            # Build a human-readable contribution table
            contrib_lines = []
            for feat, info in feature_contributions.items():
                contrib_lines.append(
                    f"  - {feat}: value={info['this_company_value']}, "
                    f"local_contribution={info['local_contribution_pct']}%"
                )
            contrib_text = "\n".join(contrib_lines)

            prompt = f"""You are an Explainable AI (XAI) specialist. The ML model '{model_name}' scored '{company}' at {crdi_score}/100 ({risk_label}).

Below are the PER-COMPANY local feature contributions (how much each metric influenced THIS specific company's score):
{contrib_text}

Write exactly 3 sentences:
1. State which specific metric(s) dominated this company's score and by how much (use the percentages).
2. Explain WHY that metric matters forensically (e.g., high geo_shell_risk means the building at the registered address doesn't match a corporate HQ).
3. State whether the weight distribution is expected or unusual for this risk band.

Be precise, data-driven, and professional. Reference the actual percentage contributions."""

            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.0
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            return f"Failed to generate weight rationale: {e}"

    @staticmethod
    def format_report(score_result: Dict[str, Any]) -> str:
        """Pretty-print the CRDI report to console."""
        r = score_result
        lines = [
            "",
            " " * 64,
            f"  CRDI  Truth Score Report   {r['company']}",
            " " * 64,
            f"",
            f"  {r['risk_icon']}  CRDI Score: {r['crdi_score']*100:.2f} / 100.00",
            f"  Risk Level : {r['risk_label']}",
            f"  {r['risk_description']}",
            f"",
            " " * 64,
            "  Module Breakdown:",
        ]
        for b in r.get("breakdown", []):
            weight_val = b['weight']
            if isinstance(weight_val, str):
                 weight_str = f"(weight: {weight_val})"
            else:
                 weight_str = f"(weight: {weight_val:.0%})"
            
            lines.append(f"      {b['module']}: {b['score']:.2%} {weight_str}   {b['verdict']}")
            if "key_red_flags" in b.get("details", {}) and b["details"]["key_red_flags"]:
                for flag in b["details"]["key_red_flags"]:
                    lines.append(f"        {flag}")

        lines.append(" " * 64)

        # Per-company feature contribution breakdown
        if r.get("feature_contributions"):
             lines.append("  Per-Company Feature Contributions:")
             for feat, info in r["feature_contributions"].items():
                 lines.append(f"    {feat}: value={info['this_company_value']}, contribution={info['local_contribution_pct']}%")
             lines.append(" " * 64)

        # Dynamic LLM-generated weight rationale (replaces old hardcoded block)
        if r.get("weight_rationale"):
             lines.append(f"  Weighting Rationale (AI Generated): {r['weight_rationale']}")
             lines.append(" " * 64)

        if r.get("dynamic_rationale"):
             lines.append(f"  AI Rationale Summary: {r['dynamic_rationale']}")
             lines.append(" " * 64)

        lines.append(f"  Generated: {r['generated_at']}")
        lines.append(" " * 64)
        return "\n".join(lines)
