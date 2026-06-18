import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", ".*tight_layout.*")

INPUT = "../data/energy_commodity_market_data.csv"
OUTPUT_CSV = "../data/market_regime_output.csv"
OUTPUT_PNG = "../outputs/market_regimes.png"
OUTPUT_CORR = "../data/regime_correlations.csv"
N_FORECAST = 30


def regimes_labels(k):
    if k == 2:
        return ["Low-Vol (Normal)", "High-Vol (Crisis)"]
    return ["Low-Vol", "Mid-Vol", "High-Vol"]


def identify_crisis_state(smoothed, endog):
    n_regimes = smoothed.shape[1]
    variances = [
        np.average(endog ** 2, weights=smoothed.iloc[:, i])
        for i in range(n_regimes)
    ]
    crisis_idx = int(np.argmax(variances))
    return crisis_idx, variances


def regime_forecast(transition_matrix, current_probs, steps):
    probs = np.array(current_probs)
    forecasts = [probs]
    for _ in range(steps):
        probs = probs @ transition_matrix
        forecasts.append(probs)
    return np.array(forecasts)


def main():
    df = pd.read_csv(INPUT, index_col="Date", parse_dates=True)

    endog = df["Brent_LogRet"].dropna().copy()
    exog_raw = df.loc[endog.index, "VIX"].shift(1)
    exog_scaled = (exog_raw - exog_raw.mean()) / exog_raw.std()
    exog_scaled.name = "VIX_lag1_scaled"

    paired = pd.concat({"e": endog, "x": exog_scaled}, axis=1).dropna()
    endog = paired["e"]
    exog_scaled = paired["x"]

    print("-" * 65)
    print("  REGIME-SWITCHING MODEL -- Markov Regression")
    print("-" * 65)
    print(f"  Endogenous: Brent Log Returns (N={len(endog)})")
    print(f"  Exogenous:  VIX (lag 1, z-scored)")

    # -- Model Comparison --------------------------------------------
    models = {}
    for k in [2, 3]:
        try:
            m = MarkovRegression(
                endog, k_regimes=k, trend="c", exog=exog_scaled,
                switching_variance=True,
            )
            f = m.fit(search_reps=20, disp=False)
            models[k] = f
        except Exception as e:
            print(f"  {k}-state model failed: {e}")

    if not models:
        print("  ERROR: No models converged. Aborting.")
        return

    best_k = min(models, key=lambda k: models[k].aic)
    fit = models[best_k]

    print(f"\n  Model Selection:")
    for k, f in sorted(models.items()):
        marker = " <-- SELECTED" if k == best_k else ""
        print(f"    {k}-state:  AIC={f.aic:>8.2f}  BIC={f.bic:>8.2f}{marker}")

    # -- Fallback: if selected model has degenerate crisis, force 2-state --
    tmp_smoothed = fit.smoothed_marginal_probabilities
    tmp_idx, _ = identify_crisis_state(tmp_smoothed, endog)
    crisis_obs = (tmp_smoothed.iloc[:, tmp_idx] > 0.5).sum()
    if crisis_obs < 30 and 2 in models:
        print(f"\n  WARNING: Crisis regime has only {crisis_obs} obs -- using 2-state model instead")
        best_k = 2
        fit = models[2]

    # -- Regime Assignment -------------------------------------------
    smoothed = fit.smoothed_marginal_probabilities
    crisis_idx, variances = identify_crisis_state(smoothed, endog)
    labels = regimes_labels(best_k)

    print(f"\n  Regime Variances:")
    for i, v in enumerate(variances):
        ann_vol = np.sqrt(v) * np.sqrt(252)
        label = labels[i]
        marker = " <-- CRISIS" if i == crisis_idx else ""
        print(f"    {label:<24} sigma^2={v:.8f}  ann.vol={ann_vol:>8.2%}{marker}")

    # -- Transition Matrix & Expected Duration -----------------------
    P_matrix = np.zeros((best_k, best_k))
    if "transition_matrix" in fit.params.index:
        raw = fit.params["transition_matrix"]
        P_matrix = np.array(raw).reshape(best_k, best_k)
    else:
        for i in range(best_k):
            for j in range(best_k):
                key = f"p[{i}->{j}]"
                if key in fit.params.index:
                    P_matrix[i, j] = fit.params[key]
    durations = 1 / (1 - np.diag(P_matrix))

    print(f"\n  Transition Matrix ({best_k}x{best_k}):")
    for i, row in enumerate(P_matrix):
        label = labels[i]
        print(f"    {label:<24} " + "  ".join(f"{p:.4f}" for p in row))
    print(f"\n  Expected Regime Duration (days):")
    for i, d in enumerate(durations):
        print(f"    {labels[i]:<24} {d:.0f} days (~{d/21:.1f} months)")

    # -- Residual Diagnostics ----------------------------------------
    residuals = fit.resid.dropna()
    lb = acorr_ljungbox(residuals, lags=[5, 10, 21], return_df=True)
    arch_result = het_arch(residuals, nlags=10)
    arch_stat, arch_p = arch_result[0], arch_result[1]
    jb_stat, jb_p = stats.jarque_bera(residuals)

    sections = {
        "Residual Diagnostics": [
            ("Ljung-Box (lag 5)",  f"stat={lb.loc[5, 'lb_stat']:.2f}  p={lb.loc[5, 'lb_pvalue']:.4f}"),
            ("Ljung-Box (lag 10)", f"stat={lb.loc[10, 'lb_stat']:.2f}  p={lb.loc[10, 'lb_pvalue']:.4f}"),
            ("Ljung-Box (lag 21)", f"stat={lb.loc[21, 'lb_stat']:.2f}  p={lb.loc[21, 'lb_pvalue']:.4f}"),
            ("ARCH(10)",           f"stat={arch_stat:.2f}  p={arch_p:.4f}"),
            ("Jarque-Bera",        f"stat={jb_stat:.2f}  p={jb_p:.4f}"),
        ],
    }

    print(f"\n  -- Residual Diagnostics --")
    for section, tests in sections.items():
        for name, result in tests:
            print(f"  {name:<28} {result}")

    # -- Regime-Conditional Correlations -----------------------------
    prob_col = "High_Vol_Probability"
    df[prob_col] = np.nan
    df.loc[endog.index, prob_col] = smoothed[crisis_idx]

    assets_ret = {
        "Brent_LogRet": df["Brent_LogRet"],
        "WTI_LogRet": df["WTI_LogRet"],
        "Gold_LogRet": df["Gold_LogRet"],
        "VIX_Ret": df["VIX"].pct_change(),
    }
    ret_df = pd.DataFrame(assets_ret).dropna()
    aligned = ret_df.loc[endog.index]

    norm_mask = smoothed[crisis_idx].values <= 0.5
    crisis_mask = smoothed[crisis_idx].values > 0.5
    aligned_norm = aligned.iloc[norm_mask]
    aligned_crisis = aligned.iloc[crisis_mask]

    n_norm = norm_mask.sum()
    n_crisis = crisis_mask.sum()
    corr_normal = aligned_norm.corr()
    corr_crisis = aligned_crisis.corr()

    print(f"\n  -- Regime-Conditional Correlations --")
    print(f"\n  Normal Regime ({n_norm} obs):")
    print(corr_normal.to_string(float_format=lambda x: f"{x:+.4f}"))
    print(f"\n  Crisis Regime ({n_crisis} obs):")
    print(corr_crisis.to_string(float_format=lambda x: f"{x:+.4f}"))

    corr_normal.to_csv(OUTPUT_CORR.replace(".csv", "_normal.csv"))
    corr_crisis.to_csv(OUTPUT_CORR.replace(".csv", "_crisis.csv"))

    # -- Forward Regime Forecast -------------------------------------
    last_probs = smoothed.iloc[-1, crisis_idx]
    current = np.array([smoothed.iloc[-1, i] for i in range(best_k)])
    forecast = regime_forecast(P_matrix, current, N_FORECAST)

    print(f"\n  -- Forward Crisis Probability Forecast --")
    print(f"  Current:  {current[crisis_idx]:.4f}")
    for step in [5, 10, 21, 30]:
        idx = min(step, len(forecast) - 1)
        print(f"  T+{step:>2}d:   {forecast[idx, crisis_idx]:.4f}")

    # -- Add regime output to DataFrame ------------------------------
    df["Predicted_Regime"] = (df[prob_col] > 0.5).astype(int)
    df.loc[endog.index, "Regime_Residual"] = residuals.values
    df.loc[endog.index, "Regime_Residual_Sq"] = residuals.values ** 2

    # -- Enhanced Plot ----------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1.2, 0.8, 0.8], hspace=0.25)

    # Panel 1 -- Normalized Prices with Regime Shading
    ax1 = fig.add_subplot(gs[0])
    ref_price = df.loc[endog.index[0]]
    brent_base = df["Brent Crude"] / ref_price["Brent Crude"] * 100
    gold_base = df["Gold"] / ref_price["Gold"] * 100

    ax1.plot(df.index, brent_base, color="#1f77b4", lw=0.9, label="Brent Crude (Base 100)")
    ax1.plot(df.index, gold_base, color="#ff9800", lw=0.9, label="Gold (Base 100)")
    valid = df[prob_col].notna()
    ax1.fill_between(
        df.index[valid], brent_base.min(), brent_base.max(),
        where=df.loc[valid, prob_col] > 0.5,
        color="red", alpha=0.15, label="Crisis Regime (P > 0.5)",
    )
    ax1.set_ylabel("Normalized Price (Base 100)", fontsize=10)
    ax1.set_title(
        f"Energy & Commodity Benchmarks -- {best_k}-State Markov Regime-Switching Model",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Panel 2 -- Regime Probability + Forward Forecast
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df.index[valid], df.loc[valid, prob_col],
             color="#d62728", lw=1.0, label="Smoothed P(Crisis)")
    ax2.axhline(0.5, color="gray", lw=0.6, ls="--", alpha=0.5)

    fwd_dates = pd.date_range(df.index[-1], periods=len(forecast), freq="B")
    ax2.plot(fwd_dates, forecast[:, crisis_idx],
             color="#d62728", lw=0.8, ls="--", alpha=0.7,
             label=f"Forward Forecast ({N_FORECAST} days)")

    ax2.fill_between(
        fwd_dates[:2], 0, 1,
        color="red", alpha=0.08,
    )
    ax2.set_ylabel("Crisis Probability", fontsize=10)
    ax2.set_ylim(-0.02, 1.08)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Panel 3 -- Residuals
    ax3 = fig.add_subplot(gs[2])
    res_plot = residuals.iloc[-252:] if len(residuals) > 252 else residuals
    ax3.bar(res_plot.index, res_plot.values, width=1.0,
            color="#1f77b4", alpha=0.6, edgecolor="none")
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.set_ylabel("Residuals", fontsize=10)
    ax3.set_title("Model Residuals (last 252 obs)", fontsize=10)
    ax3.grid(True, alpha=0.2)

    # Panel 4 -- Residual Squared (volatility clustering check)
    ax4 = fig.add_subplot(gs[3])
    res_sq = residuals.iloc[-252:] ** 2 if len(residuals) > 252 else residuals ** 2
    ax4.bar(res_sq.index, res_sq.values, width=1.0,
            color="#d62728", alpha=0.5, edgecolor="none")
    ax4.set_ylabel("Squared Residuals", fontsize=10)
    ax4.set_xlabel("Date", fontsize=10)
    ax4.set_title("Volatility Clustering (residuals^2)", fontsize=10)
    ax4.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"\n  Chart saved: {OUTPUT_PNG}")

    df.to_csv(OUTPUT_CSV)
    print(f"  Data saved: {OUTPUT_CSV}  ({len(df)} rows)")
    print()
    print(fit.summary())


if __name__ == "__main__":
    main()
