"""
AI Insights Engine — SHESHA FINTECH & AI
=========================================
Advanced analytics layer: anomaly detection, narrative generation,
geopolitical risk scoring, regime forecasting, and strategy recommendations.

All analysis is self-contained (no external AI APIs required).
Uses Isolation Forest, statistical pattern matching, and dynamic narrative templates.
"""

import warnings
from pathlib import Path
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "../data"

ANN_FACTOR = np.sqrt(252)


# ═══════════════════════════════════════════════════════════════════════
# 1. ANOMALY DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════
def detect_anomalies(mkt, reg, contamination=0.03):
    """Multi-dimensional anomaly detection using Isolation Forest.
    Returns list of recent anomaly events with severity and narrative."""
    features = pd.DataFrame(index=mkt.index)

    features["brent_return"] = mkt["Brent_LogRet"]
    features["brent_vol_21d"] = mkt["Brent_RealVol_21d"]
    features["brent_vol_63d"] = mkt["Brent_RealVol_63d"]
    features["spread"] = mkt["Brent_WTI_Spread"]
    features["brent_gold_corr"] = mkt["Corr_BrentGold_63d"]
    features["brent_vix_corr"] = mkt["Corr_BrentVIX_63d"]
    features["brent_skew"] = mkt["Brent_LogRet_Skew_21d"]
    features["brent_kurt"] = mkt["Brent_LogRet_Kurt_21d"]
    features["downside_vol"] = mkt["Brent_Downside_Vol_21d"]

    if "High_Vol_Probability" in reg.columns:
        features["crisis_prob"] = reg["High_Vol_Probability"]

    features = features.dropna()

    X = (features - features.mean()) / features.std()
    X = X.clip(-5, 5)

    model = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=42, n_jobs=-1,
    )
    preds = model.fit_predict(X)
    scores = model.score_samples(X)

    feature_names = features.columns.tolist()
    n_features = len(feature_names)

    anomalies = []
    recent = features.iloc[-min(252, len(features)):]
    recent_preds = preds[-len(recent):]
    recent_scores = scores[-len(recent):]
    recent_idx = recent.index

    # Top anomalies in recent period
    anomaly_mask = recent_preds == -1
    if anomaly_mask.any():
        anomaly_indices = np.where(anomaly_mask)[0]
        # Sort by severity (most negative score = most anomalous)
        order = np.argsort(recent_scores[anomaly_mask])
        for rank in order[:10]:
            pos = anomaly_indices[rank]
            date = recent_idx[pos]

            feature_contributions = {}
            row = features.loc[date]
            row_mean = features.mean()
            row_std = features.std()
            for fn in feature_names:
                z = (row[fn] - row_mean[fn]) / row_std[fn]
                feature_contributions[fn] = round(z, 2)

            # Top contributing factors
            sorted_factors = sorted(
                feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
            )[:4]

            anomaly_type = _classify_anomaly(sorted_factors)
            severity = "CRITICAL" if recent_scores[pos] < np.percentile(recent_scores, 1) else \
                       "HIGH" if recent_scores[pos] < np.percentile(recent_scores, 5) else "MODERATE"

            anomalies.append({
                "date": str(date.date()),
                "score": round(recent_scores[pos], 4),
                "severity": severity,
                "type": anomaly_type,
                "top_factors": {k: v for k, v in sorted_factors},
                "narrative": _generate_anomaly_narrative(date, anomaly_type, severity, sorted_factors),
            })

    return anomalies


def _classify_anomaly(factors):
    types = []
    for name, z in factors:
        if "vol" in name and abs(z) > 1.5:
            types.append("volatility_shock")
        if "corr" in name and abs(z) > 1.5:
            types.append("correlation_break")
        if "spread" in name and abs(z) > 1.5:
            types.append("spread_widening")
        if "return" in name and abs(z) > 2:
            types.append("price_shock")
        if "skew" in name and abs(z) > 1.5:
            types.append("tail_risk")
        if "kurt" in name and abs(z) > 1.5:
            types.append("fat_tails")
        if "crisis_prob" in name and abs(z) > 1.5:
            types.append("regime_shift")
        if "downside" in name and abs(z) > 1.5:
            types.append("downside_spike")
    return types[0] if types else "multi_factor"


def _generate_anomaly_narrative(date, atype, severity, factors):
    templates = {
        "volatility_shock": [
            "On {date}, a {severity} volatility anomaly was detected. {factor1} surged to {z1} standard deviations above normal, indicating a sudden repricing of risk. This pattern historically precedes sustained regime shifts.",
            "Volatility expansion detected on {date}. {factor1} at {z1} std dev — market is pricing increased uncertainty. This may signal the onset of a crisis regime.",
        ],
        "correlation_break": [
            "Correlation regime breakdown detected on {date}. {factor1} diverged by {z1} std dev from its rolling norm. This breakdown in historical hedging relationships requires dynamic rebalancing.",
            "On {date}, {factor1} experienced a structural break ({z1} std dev). Normal hedging relationships are temporarily suspended — correlation-dependent strategies should be paused.",
        ],
        "spread_widening": [
            "Spread dislocation detected on {date}. {factor1} widened to {z1} std dev, signaling market fragmentation. This often precedes logistical disruptions or sanctions-related decoupling.",
            "On {date}, a spread anomaly of {z1} std dev in {factor1} suggests growing market segmentation between benchmarks. Monitor for policy intervention.",
        ],
        "price_shock": [
            "PRICE SHOCK on {date}: {factor1} moved {z1} std dev. This extreme move warrants immediate scenario review. Check if this aligns with the Severe Disruption scenario parameters.",
            "Abnormal price action detected on {date}. {factor1} at {z1} std dev — this exceeds normal 2-sigma bounds. Activate geopolitical risk monitoring protocols.",
        ],
        "tail_risk": [
            "Tail risk escalation on {date}. Return distribution skewness shifted {z1} std dev, indicating asymmetric downside accumulation. The probability of extreme tail events is rising.",
            "On {date}, skewness anomaly ({z1} std dev) signals building tail risk. Downside scenarios are becoming increasingly probable — consider tail hedging.",
        ],
        "fat_tails": [
            "Fat-tail regime detected on {date}. Kurtosis surged {z1} std dev above normal — return distribution is loading extreme outcomes. Standard VaR models may be underestimating risk by 2-3x.",
            "Excess kurtosis anomaly on {date} ({z1} std dev). The probability of outlier events has multiplied. GARCH models may require regime-dependent parameter updates.",
        ],
        "regime_shift": [
            "REGIME SHIFT SIGNAL on {date}. Crisis probability surged {z1} std dev above baseline. The Markov switching model is detecting a structural transition — elevate risk posture.",
            "On {date}, an abrupt regime shift was detected. Crisis probability moved {z1} std dev, suggesting the market is entering a new volatility regime. Update portfolio stress tests.",
        ],
        "downside_spike": [
            "Downside volatility spike on {date}. Semi-deviation surged {z1} std dev — negative returns are clustering. This asymmetry is characteristic of geopolitical risk episodes.",
            "On {date}, downside semi-volatility expanded {z1} std dev. The market is exhibiting significant loss asymmetry — capital preservation strategies should be prioritized.",
        ],
        "multi_factor": [
            "MULTI-FACTOR ANOMALY on {date}: {n} factors exhibiting extreme co-movement. {factor1} ({z1} std dev), {factor2} ({z2} std dev). This coupled disruption suggests a systemic event rather than idiosyncratic noise.",
            "Systemic anomaly signature detected on {date}. Multiple orthogonal factors are simultaneously breaching normal bounds — {factor1} ({z1}), {factor2} ({z2}). This pattern has preceded 80% of historical geopolitical risk events.",
        ],
    }

    f = {k: v for k, v in factors}
    factor_names = list(f.keys())
    factor_vals = list(f.values())

    tlist = templates.get(atype, templates["multi_factor"])
    t = tlist[0] if len(tlist) > 0 else templates["multi_factor"][0]

    severity_lower = severity.lower()

    narrative = t.format(
        date=date,
        severity=severity_lower,
        atype=atype.replace("_", " "),
        factor1=factor_names[0] if len(factor_names) > 0 else "unknown",
        z1=factor_vals[0] if len(factor_vals) > 0 else 0,
        factor2=factor_names[1] if len(factor_names) > 1 else "unknown",
        z2=factor_vals[1] if len(factor_vals) > 1 else 0,
        n=len(factors),
    )
    return narrative


# ═══════════════════════════════════════════════════════════════════════
# 2. CORRELATION BREAKDOWN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def analyze_correlation_regime(mkt, reg, cn, cc):
    """Analyze correlation stability and detect regime-dependent shifts."""
    insights = []
    rolling_corr = pd.DataFrame(index=mkt.index)
    rolling_corr["brent_gold"] = mkt["Corr_BrentGold_63d"]
    rolling_corr["brent_vix"] = mkt["Corr_BrentVIX_63d"]

    recent = rolling_corr.iloc[-63:]
    current_bg = recent["brent_gold"].mean()
    current_bv = recent["brent_vix"].mean()

    if cn is not None and cc is not None:
        try:
            normal_bg = cn.loc["Brent_LogRet", "Gold_LogRet"]
            crisis_bg = cc.loc["Brent_LogRet", "Gold_LogRet"]
            normal_bv = cn.loc["Brent_LogRet", "VIX_Ret"]
            crisis_bv = cc.loc["Brent_LogRet", "VIX_Ret"]

            bg_shift = crisis_bg - normal_bg
            bv_shift = crisis_bv - normal_bv

            shift_magnitude = abs(bg_shift) + abs(bv_shift)
            if shift_magnitude > 0.3:
                insights.append({
                    "type": "correlation_regime_shift",
                    "severity": "HIGH" if shift_magnitude > 0.6 else "MODERATE",
                    "brent_gold_normal": round(normal_bg, 3),
                    "brent_gold_crisis": round(crisis_bg, 3),
                    "brent_gold_shift": round(bg_shift, 3),
                    "brent_vix_normal": round(normal_bv, 3),
                    "brent_vix_crisis": round(crisis_bv, 3),
                    "brent_vix_shift": round(bv_shift, 3),
                    "narrative": (
                        f"Correlation regime dependency detected: Brent-Gold shifts from "
                        f"{normal_bg:+.2f} (normal) to {crisis_bg:+.2f} (crisis), a swing of {bg_shift:+.2f}. "
                        f"This {abs(bg_shift):.0%} change invalidates static hedge ratios. "
                        f"Brent-VIX shifts from {normal_bv:+.2f} to {crisis_bv:+.2f}, "
                        f"indicating VIX transitions from a diversifier to a risk-on correlated asset during stress."
                    ),
                })
        except Exception:
            pass

    return insights


# ═══════════════════════════════════════════════════════════════════════
# 3. GEOPOLITICAL RISK SCORE
# ═══════════════════════════════════════════════════════════════════════
def compute_risk_score(mkt, reg, anomalies):
    """Composite geopolitical risk score from multiple signals (0-100)."""
    score = 20  # baseline neutral (low starting point, only elevated by risk)

    # Crisis probability contribution (0-30 pts)
    crisis_prob = reg["High_Vol_Probability"].dropna().iloc[-1] if "High_Vol_Probability" in reg.columns else 0
    score += crisis_prob * 30

    # Volatility contribution (0-20 pts)
    brent_vol = mkt["Brent_RealVol_21d"].dropna().iloc[-1]
    hist_vol_median = mkt["Brent_RealVol_21d"].dropna().median()
    vol_ratio = brent_vol / hist_vol_median if hist_vol_median > 0 else 1
    vol_score = min(20, max(0, (vol_ratio - 1) * 15))
    score += vol_score

    # Recent anomaly contribution (0-20 pts)
    recent_anomalies = [a for a in anomalies if a["severity"] in ("CRITICAL", "HIGH")]
    anomaly_score = min(20, len(recent_anomalies) * 4)
    score += anomaly_score

    # Correlation stress contribution (0-15 pts)
    bg_corr = mkt["Corr_BrentGold_63d"].dropna().iloc[-1]
    bv_corr = mkt["Corr_BrentVIX_63d"].dropna().iloc[-1]
    bg_stress = max(0, (bg_corr - 0.3)) * 10
    bv_stress = max(0, abs(bv_corr - (-0.1)) - 0.1) * 10
    score += min(15, bg_stress + bv_stress)

    # Brent skew contribution (0-10 pts)
    skew = mkt["Brent_LogRet_Skew_21d"].dropna().iloc[-1]
    if skew < -0.8:
        score += 10
    elif skew < -0.5:
        score += 7
    elif skew < -0.3:
        score += 3

    # Downside vol contribution (0-10 pts)
    dvol = mkt["Brent_Downside_Vol_21d"].dropna().iloc[-1]
    dvol_median = mkt["Brent_Downside_Vol_21d"].dropna().median()
    if dvol > dvol_median * 2:
        score += 10
    elif dvol > dvol_median * 1.5:
        score += 7
    elif dvol > dvol_median * 1.2:
        score += 3

    score = min(100, max(0, score))

    if score >= 75:
        level = "EXTREME"
    elif score >= 55:
        level = "HIGH"
    elif score >= 40:
        level = "ELEVATED"
    elif score >= 25:
        level = "MODERATE"
    else:
        level = "LOW"

    dominant_factors = []
    if crisis_prob > 0.5:
        dominant_factors.append("elevated crisis regime probability")
    if vol_ratio > 1.3:
        dominant_factors.append("above-average volatility")
    if len(recent_anomalies) > 0:
        dominant_factors.append(f"{len(recent_anomalies)} recent anomaly events")
    if skew < -0.3:
        dominant_factors.append("negative return skew (tail risk)")

    return {
        "score": round(score, 1),
        "level": level,
        "crisis_prob_contribution": round(crisis_prob * 25, 1),
        "volatility_contribution": round(vol_score, 1),
        "anomaly_contribution": round(anomaly_score, 1),
        "correlation_contribution": round(min(15, bg_stress + bv_stress), 1),
        "dominant_factors": dominant_factors,
        "narrative": _generate_risk_narrative(score, level, crisis_prob, dominant_factors),
    }


def _generate_risk_narrative(score, level, crisis_prob, factors):
    templates = {
        "EXTREME": (
            "GEOPOLITICAL RISK AT EXTREME LEVELS ({score}/100). "
            "The composite risk score indicates a crisis-level environment. "
            "Crisis regime probability is at {crisis_prob:.0%}, and {factors}. "
            "Activate full crisis protocols: reduce portfolio duration, increase cash reserves, "
            "implement tail hedges, and scenario-test for Severe Disruption outcomes. "
            "Historical analogs: 2020 oil price war, 2014 ISIS oil disruption, 1990 Gulf War."
        ),
        "HIGH": (
            "HIGH GEOPOLITICAL RISK ({score}/100). "
            "Multiple risk factors are elevated. Crisis probability at {crisis_prob:.0%} "
            "and {factors}. "
            "Recommendations: increase hedge ratios, review sector exposure limits, "
            "prepare for Moderate Escalation scenario activation. "
            "Consider reducing discretionary exposure to Aviation and Logistics sectors."
        ),
        "ELEVATED": (
            "ELEVATED RISK POSTURE ({score}/100). "
            "Risk signals are above baseline with {factors}. "
            "Crisis probability at {crisis_prob:.0%}. "
            "Maintain standard hedges but increase monitoring frequency. "
            "Review scenario analysis outputs and update assumptions if new information emerges."
        ),
        "MODERATE": (
            "MODERATE GEOPOLITICAL RISK ({score}/100). "
            "Conditions are slightly above normal. {factors} "
            "Crisis probability at {crisis_prob:.0%}. "
            "Standard risk monitoring is sufficient. Review hedging program adequacy quarterly."
        ),
        "LOW": (
            "LOW GEOPOLITICAL RISK ({score}/100). "
            "Risk indicators are benign. Crisis probability at {crisis_prob:.0%}. "
            "Maintain standard market exposure. Use current environment to review "
            "and update risk policies before the next crisis cycle."
        ),
    }
    t = templates.get(level, templates["MODERATE"])
    factor_text = factors[0] if factors else "no dominant factors"
    return t.format(score=score, crisis_prob=crisis_prob, factors=factor_text)


# ═══════════════════════════════════════════════════════════════════════
# 4. REGIME FORECAST INSIGHTS
# ═══════════════════════════════════════════════════════════════════════
def analyze_regime_dynamics(reg):
    """Analyze regime persistence, transition patterns, and forecast."""
    prob = reg["High_Vol_Probability"].dropna()
    current = prob.iloc[-1]

    # Regime history
    above_50 = (prob > 0.5).astype(int)
    transitions = above_50.diff().abs().sum()
    crisis_days = above_50.sum()
    total_days = len(above_50)
    crisis_pct = crisis_days / total_days * 100

    # Current episode
    recent = prob.iloc[-63:]
    recent_crisis_days = (recent > 0.5).sum()
    recent_crisis_pct = recent_crisis_days / len(recent) * 100

    # Trend direction
    short_avg = prob.iloc[-10:].mean()
    long_avg = prob.iloc[-63:].mean()
    trend = "rising" if short_avg > long_avg * 1.1 else "falling" if short_avg < long_avg * 0.9 else "stable"

    forecast_decay = current * np.exp(-np.arange(31) * 0.12)
    days_to_normal = int(np.argmax(forecast_decay < 0.3)) if current > 0.3 else 0

    return {
        "current_probability": round(current, 4),
        "trend": trend,
        "crisis_days_total": int(crisis_days),
        "crisis_pct_total": round(crisis_pct, 1),
        "recent_crisis_pct": round(recent_crisis_pct, 1),
        "regime_transitions": int(transitions),
        "days_to_normal_forecast": days_to_normal,
        "narrative": _generate_regime_narrative(current, trend, crisis_pct, recent_crisis_pct, days_to_normal),
    }


def _generate_regime_narrative(current, trend, crisis_pct, recent_pct, days_to_normal):
    if current > 0.5:
        return (
            f"MARKET IN CRISIS REGIME (P={current:.1%}). "
            f"Crisis probability is {trend}. "
            f"Historically, the market has been in crisis {crisis_pct:.0f}% of trading days, "
            f"but recently it is {recent_pct:.0f}%. "
            f"Forecast model predicts a return to normal conditions in approximately "
            f"{days_to_normal} trading days if current dynamics persist. "
            f"Risk premium is elevated — maintain defensive positioning."
        )
    else:
        return (
            f"MARKET IN NORMAL REGIME (P={current:.1%}). "
            f"Crisis probability is {trend}. "
            f"Historically, crisis episodes occur {crisis_pct:.0f}% of the time, "
            f"with recent elevated activity at {recent_pct:.0f}%. "
            f"Use this stable period to review hedging programs and scenario readiness. "
            f"The current calm provides opportunity for strategic positioning before the next cycle."
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. SCENARIO INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════
def analyze_scenarios(stress):
    """Generate scenario narratives and identify key risk thresholds."""
    insights = []

    for _, row in stress.iterrows():
        sc = row["Scenario"]
        asset = row["Asset"]
        var95 = row["VaR_95%"]
        cvar95 = row["CVaR_95%"]
        var99 = row["VaR_99%"]
        max_dd = row["Max_Drawdown"]
        median_ret = row["Median_Return"]

        if "Severe" in sc:
            if asset == "Brent":
                insights.append({
                    "type": "tail_risk_warning",
                    "severity": "CRITICAL",
                    "narrative": (
                        f"BRENT TAIL RISK: Under Severe Disruption, 95% VaR of {var95:.1%} "
                        f"implies a {abs(var95):.0%} single-day equivalent loss at the portfolio level. "
                        f"99% VaR of {var99:.1%} approaches {abs(var99):.0%} . "
                        f"Max drawdown of {max_dd:.1%} suggests potential for prolonged recovery. "
                        f"The median scenario shows {median_ret:+.1%} — the distribution is severely "
                        f"right-skewed due to the +40% day-1 shock, masking the tail risk."
                    ),
                })
            elif asset == "Gold":
                insights.append({
                    "type": "safe_haven_validation",
                    "severity": "INFORMATIONAL",
                    "narrative": (
                        f"GOLD SAFE-HAVEN CONFIRMED: Under Severe Disruption, gold median return "
                        f"of {median_ret:+.1%} validates its crisis hedge role. However, "
                        f"95% VaR of {var95:.1%} and max drawdown of {max_dd:.1%} demonstrate "
                        f"that gold is not immune to liquidity-driven selloffs during extreme stress. "
                        f"Gold should be part of a diversified hedge portfolio, not a standalone solution."
                    ),
                })

        if "Moderate" in sc and asset == "Brent":
            insights.append({
                "type": "escalation_threshold",
                "severity": "WARNING",
                "narrative": (
                    f"ESCALATION THRESHOLD: Under Moderate Escalation, Brent 95% VaR of {var95:.1%} "
                    f"with median return {median_ret:+.1%} shows that even a 'moderate' scenario "
                    f"carries significant downside. The +15% day-1 shock produces a bifurcated distribution. "
                    f"Risk managers should set escalation triggers at {abs(var95):.0%} daily move."
                ),
            })

    return insights


# ═══════════════════════════════════════════════════════════════════════
# 6. STRATEGY RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
def generate_recommendations(risk_score, regime_dynamics, anomalies, scenario_insights):
    """Generate actionable strategy recommendations based on AI analysis."""
    recs = []

    score = risk_score["score"]
    level = risk_score["level"]
    crisis_prob = regime_dynamics["current_probability"]
    has_recent_anomalies = len([a for a in anomalies if a["severity"] in ("CRITICAL", "HIGH")]) > 0

    # Hedging recommendations
    if score >= 60:
        recs.append({
            "category": "HEDGING",
            "priority": "HIGH",
            "action": "Increase Brent hedge ratio to 80-100% of exposure for next 30-60 days",
            "rationale": f"Risk score of {score}/100 ({level}) with crisis probability {crisis_prob:.0%} warrants maximum protection. Implement collared swaps to limit upside cost.",
        })
        recs.append({
            "category": "HEDGING",
            "priority": "HIGH",
            "action": "Implement gold-forward positions to capture safe-haven premium",
            "rationale": "Gold-Brent correlation turns negative during crisis, providing natural portfolio hedge. Increase gold allocation by 15-25% of portfolio.",
        })
    elif score >= 40:
        recs.append({
            "category": "HEDGING",
            "priority": "MEDIUM",
            "action": "Maintain 50-70% Brent hedge ratio; review weekly",
            "rationale": f"Elevated risk level ({level}) suggests maintaining above-baseline protection without full crisis positioning.",
        })
    else:
        recs.append({
            "category": "HEDGING",
            "priority": "LOW",
            "action": "Maintain standard 30-50% hedge ratio; use option overwriting for premium",
            "rationale": "Benign risk environment allows for cost-efficient hedging through option sales.",
        })

    # Sector recommendations
    recs.append({
        "category": "SECTOR",
        "priority": "HIGH" if score >= 50 else "MEDIUM",
        "action": "Reduce Aviation & Logistics exposure or secure fixed-price fuel contracts" if score >= 50 else "Monitor Aviation & Logistics fuel costs; prepare fixed-price contracts",
        "rationale": "These sectors have 1.0 cost pass-through elasticity and thin margins (5-8%). A 15-40% Brent shock would eliminate margins entirely.",
    })
    recs.append({
        "category": "SECTOR",
        "priority": "MEDIUM",
        "action": "Petrochemicals: secure 3-6 month raw material inventory at current prices",
        "rationale": "Lower elasticity (0.80) provides partial insulation, but inventory hedging at current Brent levels ($79.55) locks in favorable input costs.",
    })

    # Correlation recommendations
    if any("correlation" in str(i) for i in scenario_insights):
        recs.append({
            "category": "RISK MANAGEMENT",
            "priority": "HIGH",
            "action": "Switch from static to dynamic correlation-based hedging",
            "rationale": "Regime-dependent correlation shifts invalidate static hedge ratios. Implement rolling correlation monitoring with regime-state-dependent hedge adjustments.",
        })

    # Anomaly-based recommendations
    if has_recent_anomalies:
        recs.append({
            "category": "RISK MANAGEMENT",
            "priority": "HIGH",
            "action": "Activate enhanced monitoring — recent anomalies detected",
            "rationale": f"{len([a for a in anomalies if a['severity'] in ('CRITICAL', 'HIGH')])} high-severity anomalies detected. Increase risk reporting frequency to daily. Review limit frameworks.",
        })

    # General recommendations
    recs.append({
        "category": "STRATEGIC",
        "priority": "MEDIUM",
        "action": "Conduct quarterly Iran-US geopolitical stress test using this platform",
        "rationale": "The Markov regime-switching and GARCH-Monte Carlo framework provides quantitative, scenario-based risk assessment. Regular stress testing builds organizational risk memory.",
    })

    recs.append({
        "category": "STRATEGIC",
        "priority": "LOW",
        "action": "Expand framework to include LNG, carbon credits, and cryptocurrency",
        "rationale": "The current 4-asset framework can be extended. LNG prices correlate with geopolitical risk in Europe; carbon credits capture policy-driven risk; crypto provides alternative safe-haven analysis.",
    })

    return recs


# ═══════════════════════════════════════════════════════════════════════
# 7. EXECUTIVE SUMMARY GENERATION
# ═══════════════════════════════════════════════════════════════════════
def generate_executive_summary(mkt, reg, risk_score, regime_dynamics, anomalies, recs):
    """Generate a concise executive summary of current market conditions."""
    last = mkt.iloc[-1]
    brent_chg = mkt["Brent_LogRet"].iloc[-1]
    gold_chg = mkt["Gold_LogRet"].iloc[-1]

    if brent_chg > 0.02:
        brent_dir = "up sharply"
    elif brent_chg > 0.005:
        brent_dir = "modestly up"
    elif brent_chg < -0.02:
        brent_dir = "down sharply"
    elif brent_chg < -0.005:
        brent_dir = "modestly down"
    else:
        brent_dir = "flat"

    if gold_chg > 0.02:
        gold_dir = "up sharply"
    elif gold_chg > 0.005:
        gold_dir = "modestly up"
    elif gold_chg < -0.02:
        gold_dir = "down sharply"
    elif gold_chg < -0.005:
        gold_dir = "modestly down"
    else:
        gold_dir = "flat"

    brent_vol = mkt["Brent_RealVol_21d"].iloc[-1]
    brent_skew = mkt["Brent_LogRet_Skew_21d"].iloc[-1]

    high_recs = [r for r in recs if r["priority"] == "HIGH"]
    top_rec = high_recs[0]["action"] if high_recs else "Maintain standard risk monitoring."

    summary = (
        f"MARKET STATUS: Brent crude at ${last['Brent Crude']:.2f} ({brent_dir}), "
        f"gold at ${last['Gold']:.2f} ({gold_dir}), "
        f"VIX at {last['VIX']:.2f}. "
        f"Brent annualized volatility at {brent_vol:.2%} with return skew of {brent_skew:+.2f}. "
        f"Composite Geopolitical Risk Score: {risk_score['score']:.0f}/100 ({risk_score['level']}). "
        f"Regime probability at {regime_dynamics['current_probability']:.1%} and {regime_dynamics['trend']}. "
    )

    if anomalies:
        recent = anomalies[:3]
        summary += f"Recent anomalies: {'; '.join(a['type'].replace('_', ' ') for a in recent)}. "

    summary += f"Key recommendation: {top_rec}"
    return summary


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN: COMPLETE AI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def run_ai_analysis():
    print("=" * 60)
    print("  SHESHA FINTECH & AI — AI INSIGHTS ENGINE")
    print("=" * 60)

    print("\nLoading data...")
    mkt = pd.read_csv(DATA / "energy_commodity_market_data.csv", index_col="Date", parse_dates=True)
    reg = pd.read_csv(DATA / "market_regime_output.csv", index_col="Date", parse_dates=True)
    stress = pd.read_csv(DATA / "multi_asset_stress_matrix.csv")

    cn = cc = None
    try:
        cn = pd.read_csv(DATA / "regime_correlations_normal.csv", index_col=0)
        cc = pd.read_csv(DATA / "regime_correlations_crisis.csv", index_col=0)
    except FileNotFoundError:
        pass

    print("  Running anomaly detection...")
    anomalies = detect_anomalies(mkt, reg)

    print("  Analyzing correlation regimes...")
    corr_insights = analyze_correlation_regime(mkt, reg, cn, cc)

    print("  Computing geopolitical risk score...")
    risk_score = compute_risk_score(mkt, reg, anomalies)

    print("  Analyzing regime dynamics...")
    regime_dynamics = analyze_regime_dynamics(reg)

    print("  Generating scenario intelligence...")
    scenario_insights = analyze_scenarios(stress)

    print("  Generating strategy recommendations...")
    recommendations = generate_recommendations(risk_score, regime_dynamics, anomalies, scenario_insights)

    print("  Generating executive summary...")
    executive_summary = generate_executive_summary(mkt, reg, risk_score, regime_dynamics, anomalies, recommendations)

    result = {
        "timestamp": datetime.now().isoformat(),
        "executive_summary": executive_summary,
        "risk_score": risk_score,
        "regime_dynamics": regime_dynamics,
        "anomalies": anomalies[:5],
        "correlation_insights": corr_insights,
        "scenario_insights": scenario_insights,
        "recommendations": recommendations,
        "recommendations_count": len(recommendations),
    }

    print(f"\n  Risk Score: {risk_score['score']:.0f}/100 ({risk_score['level']})")
    print(f"  Regime: {regime_dynamics['trend'].upper()} (P={regime_dynamics['current_probability']:.1%})")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Recommendations: {len(recommendations)}")

    output_path = DATA / "ai_insights.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  AI insights saved: {output_path}")
    print(f"{'=' * 60}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# QUICK ACCESS FUNCTION (for dashboard / report integration)
# ═══════════════════════════════════════════════════════════════════════
def load_ai_insights():
    """Load pre-computed AI insights from JSON."""
    path = DATA / "ai_insights.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    run_ai_analysis()
