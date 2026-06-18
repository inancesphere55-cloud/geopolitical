import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

INPUT = "../data/market_regime_output.csv"
OUTPUT_CSV = "../data/multi_asset_stress_matrix.csv"
OUTPUT_DIST = "../data/simulation_distribution.csv"
N_PATHS = 1000
HORIZON = 30
SEED = 42

SCENARIOS = {
    "A_StatusQuo": {
        "label": "A: Status Quo (State 0)",
        "p_state1": 0.0,
        "brent_shock": 0.0,
        "gold_shock": 0.0,
    },
    "B_ModEscalation": {
        "label": "B: Moderate Escalation",
        "p_state1": 0.6,
        "brent_shock": 0.15,
        "gold_shock": 0.05,
    },
    "C_SevereDisruption": {
        "label": "C: Severe Disruption",
        "p_state1": 1.0,
        "brent_shock": 0.40,
        "gold_shock": 0.20,
    },
}


def estimate_garch11(lrets):
    """MLE for GARCH(1,1): sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}"""
    vals = lrets.dropna().values
    n = len(vals)
    init_var = np.var(vals)

    def nll(theta):
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = np.full(n, init_var)
        for t in range(1, n):
            sigma2[t] = omega + alpha * vals[t - 1] ** 2 + beta * sigma2[t - 1]
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + vals ** 2 / sigma2)
        return -ll

    result = minimize(
        nll,
        [init_var * 0.05, 0.10, 0.85],
        bounds=[(1e-8, None), (1e-8, 1), (1e-8, 1)],
        method="L-BFGS-B",
    )
    return result.x


def simulate_garch_multivariate(rng, n_paths, horizon, corr_matrix,
                                mu, omega, alpha, beta,
                                shock_day1=0.0):
    """
    Simulate correlated GARCH(1,1) paths for 2 assets (Brent & Gold).
    Returns: (paths_asset1, paths_asset2) each shape (n_paths, horizon+1)
    """
    n_assets = 2
    L = np.linalg.cholesky(corr_matrix)
    paths = np.zeros((n_assets, n_paths, horizon + 1))
    sigma2 = np.ones((n_assets, n_paths))

    for t in range(1, horizon + 1):
        Z = rng.normal(0, 1, size=(n_paths, n_assets))
        Z_corr = Z @ L.T

        shock = np.zeros(n_paths)
        if t == 1:
            shock = shock_day1

        for a in range(n_assets):
            eps = mu[a] + np.sqrt(sigma2[a]) * Z_corr[:, a] + (shock if a == 0 else 0)
            paths[a, :, t] = paths[a, :, t - 1] * np.exp(eps)
            sigma2[a] = omega[a] + alpha[a] * eps ** 2 + beta[a] * sigma2[a]

    return paths[0], paths[1]


def compute_var_cvar(final_prices, start_price, conf_levels=[0.95, 0.99]):
    returns = final_prices / start_price - 1
    results = {}
    for conf in conf_levels:
        var = np.percentile(returns, (1 - conf) * 100)
        cvar = returns[returns <= var].mean()
        results[f"VaR_{conf:.0%}"] = var
        results[f"CVaR_{conf:.0%}"] = cvar
    results["Max_Drawdown"] = (final_prices / start_price - 1).min()
    results["Max_Upside"] = (final_prices / start_price - 1).max()
    return results


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(INPUT, index_col="Date", parse_dates=True)
    last_brent = df["Brent Crude"].iloc[-1]
    last_gold = df["Gold"].iloc[-1]
    print(f"  Last prices:  Brent = ${last_brent:.2f}   Gold = ${last_gold:.2f}")

    probs = df["High_Vol_Probability"].dropna()
    brent_lrets = df["Brent_LogRet"].dropna()
    gold_lrets = df["Gold_LogRet"].dropna()

    common_idx = probs.index.intersection(brent_lrets.index).intersection(gold_lrets.index)
    probs = probs.loc[common_idx]
    brent_lrets = brent_lrets.loc[common_idx]
    gold_lrets = gold_lrets.loc[common_idx]

    mask = probs > 0.5

    # -- Regime-conditional parameters -------------------------------
    mu_b0 = brent_lrets[~mask].mean()
    sigma_b0 = brent_lrets[~mask].std()
    mu_b1 = brent_lrets[mask].mean()
    sigma_b1 = brent_lrets[mask].std()

    mu_g0 = gold_lrets[~mask].mean()
    sigma_g0 = gold_lrets[~mask].std()
    mu_g1 = gold_lrets[mask].mean()
    sigma_g1 = gold_lrets[mask].std()

    corr_normal = np.corrcoef(brent_lrets[~mask], gold_lrets[~mask])
    corr_crisis = np.corrcoef(brent_lrets[mask], gold_lrets[mask])

    # Regularize correlation matrices for positive definiteness
    eps_reg = 1e-6
    corr_normal = (1 - eps_reg) * corr_normal + eps_reg * np.eye(2)
    corr_crisis = (1 - eps_reg) * corr_crisis + eps_reg * np.eye(2)

    # -- GARCH(1,1) estimation per regime ----------------------------
    print("\n  -- GARCH(1,1) Parameter Estimation --")
    garch_params = {}
    for regime_name, lrets_b, lrets_g in [
        ("Normal", brent_lrets[~mask], gold_lrets[~mask]),
        ("Crisis", brent_lrets[mask], gold_lrets[mask]),
    ]:
        if len(lrets_b) < 30:
            print(f"  {regime_name}: insufficient data, using simple vol")
            garch_params[regime_name] = {
                "omega_b": np.var(lrets_b) * 0.1,
                "alpha_b": 0.10,
                "beta_b": 0.85,
                "omega_g": np.var(lrets_g) * 0.1,
                "alpha_g": 0.10,
                "beta_g": 0.85,
                "mu_b": lrets_b.mean(),
                "mu_g": lrets_g.mean(),
            }
            continue

        omega_b, alpha_b, beta_b = estimate_garch11(lrets_b)
        omega_g, alpha_g, beta_g = estimate_garch11(lrets_g)
        annualized_vol_b = np.sqrt(omega_b / (1 - alpha_b - beta_b)) * np.sqrt(252)

        garch_params[regime_name] = {
            "omega_b": omega_b, "alpha_b": alpha_b, "beta_b": beta_b,
            "omega_g": omega_g, "alpha_g": alpha_g, "beta_g": beta_g,
            "mu_b": lrets_b.mean(), "mu_g": lrets_g.mean(),
        }
        print(f"  {regime_name:<10} Brent: omega={omega_b:.8f} alpha={alpha_b:.4f} beta={beta_b:.4f} "
              f"(ann.vol={annualized_vol_b:.2%})")
        print(f"  {regime_name:<10} Gold:  omega={omega_g:.8f} alpha={alpha_g:.4f} beta={beta_g:.4f}")

    # -- Parameter summary table -------------------------------------
    print("\n" + "-" * 75)
    print(f"{'Parameter':<30} {'State 0 (Normal)':>20} {'State 1 (Crisis)':>22}")
    print("-" * 75)
    print(f"{'Brent mean':<30} {mu_b0:>20.6f} {mu_b1:>22.6f}")
    print(f"{'Brent std':<30} {sigma_b0:>20.6f} {sigma_b1:>22.6f}")
    print(f"{'Brent ann. vol':<30} {sigma_b0*np.sqrt(252):>19.2%} {sigma_b1*np.sqrt(252):>21.2%}")
    print(f"{'Gold mean':<30} {mu_g0:>20.6f} {mu_g1:>22.6f}")
    print(f"{'Gold std':<30} {sigma_g0:>20.6f} {sigma_g1:>22.6f}")
    print(f"{'Gold ann. vol':<30} {sigma_g0*np.sqrt(252):>19.2%} {sigma_g1*np.sqrt(252):>21.2%}")
    print(f"{'Corr(Brent, Gold)':<30} {corr_normal[0,1]:>+20.4f} {corr_crisis[0,1]:>+22.4f}")
    print(f"{'Obs (aligned)':<30} {(~mask).sum():>20} {mask.sum():>22}")
    print("-" * 75)
    print()

    # -- Monte Carlo Simulation --------------------------------------
    results = []
    dist_rows = []

    for key, sc in SCENARIOS.items():
        # Weighted average parameters by scenario crisis probability
        p = sc["p_state1"]
        mu_b = (1 - p) * mu_b0 + p * mu_b1
        mu_g = (1 - p) * mu_g0 + p * mu_g1
        omega_b = (1 - p) * garch_params["Normal"]["omega_b"] + p * garch_params["Crisis"]["omega_b"]
        alpha_b = (1 - p) * garch_params["Normal"]["alpha_b"] + p * garch_params["Crisis"]["alpha_b"]
        beta_b = (1 - p) * garch_params["Normal"]["beta_b"] + p * garch_params["Crisis"]["beta_b"]
        omega_g = (1 - p) * garch_params["Normal"]["omega_g"] + p * garch_params["Crisis"]["omega_g"]
        alpha_g = (1 - p) * garch_params["Normal"]["alpha_g"] + p * garch_params["Crisis"]["alpha_g"]
        beta_g = (1 - p) * garch_params["Normal"]["beta_g"] + p * garch_params["Crisis"]["beta_g"]

        # Weighted correlation matrix
        if p > 0.5:
            corr_mat = corr_crisis
        elif p < 0.5:
            corr_mat = corr_normal
        else:
            corr_mat = (corr_normal + corr_crisis) / 2

        brent_paths = np.zeros((N_PATHS, HORIZON + 1))
        gold_paths = np.zeros((N_PATHS, HORIZON + 1))
        brent_paths[:, 0] = last_brent
        gold_paths[:, 0] = last_gold

        corr_mat = (1 - eps_reg) * corr_mat + eps_reg * np.eye(2)
        L = np.linalg.cholesky(corr_mat)
        sigma2_b = np.ones(N_PATHS) * sigma_b0 ** 2
        sigma2_g = np.ones(N_PATHS) * sigma_g0 ** 2

        shock_b = sc["brent_shock"]
        shock_g = sc["gold_shock"]

        for t in range(1, HORIZON + 1):
            in_crisis = rng.uniform(size=N_PATHS) < p
            Z = rng.normal(0, 1, size=(N_PATHS, 2))
            Z_corr = Z @ L.T

            mu_b_arr = np.where(in_crisis, mu_b1, mu_b0)
            mu_g_arr = np.where(in_crisis, mu_g1, mu_g0)
            omega_b_arr = np.where(in_crisis, garch_params["Crisis"]["omega_b"], garch_params["Normal"]["omega_b"])
            alpha_b_arr = np.where(in_crisis, garch_params["Crisis"]["alpha_b"], garch_params["Normal"]["alpha_b"])
            beta_b_arr = np.where(in_crisis, garch_params["Crisis"]["beta_b"], garch_params["Normal"]["beta_b"])
            omega_g_arr = np.where(in_crisis, garch_params["Crisis"]["omega_g"], garch_params["Normal"]["omega_g"])
            alpha_g_arr = np.where(in_crisis, garch_params["Crisis"]["alpha_g"], garch_params["Normal"]["alpha_g"])
            beta_g_arr = np.where(in_crisis, garch_params["Crisis"]["beta_g"], garch_params["Normal"]["beta_g"])

            eps_b = mu_b_arr + np.sqrt(sigma2_b) * Z_corr[:, 0]
            eps_g = mu_g_arr + np.sqrt(sigma2_g) * Z_corr[:, 1]

            if t == 1:
                eps_b += shock_b
                eps_g += shock_g

            brent_paths[:, t] = brent_paths[:, t - 1] * np.exp(eps_b)
            gold_paths[:, t] = gold_paths[:, t - 1] * np.exp(eps_g)

            sigma2_b = omega_b_arr + alpha_b_arr * eps_b ** 2 + beta_b_arr * sigma2_b
            sigma2_g = omega_g_arr + alpha_g_arr * eps_g ** 2 + beta_g_arr * sigma2_g

        # -- Distribution statistics --------------------------------
        for asset, paths in [("Brent", brent_paths), ("Gold", gold_paths)]:
            final = paths[:, -1]
            start_p = paths[0, 0]

            median = np.median(final)
            p5 = np.percentile(final, 5)
            p95 = np.percentile(final, 95)

            risk = compute_var_cvar(final, start_p, [0.90, 0.95, 0.99])

            results.append({
                "Scenario": sc["label"],
                "Asset": asset,
                "StartPrice": round(start_p, 2),
                "Median_Price": round(median, 2),
                "P5_Price": round(p5, 2),
                "P95_Price": round(p95, 2),
                "P5_Return": round(p5 / start_p - 1, 4),
                "Median_Return": round(median / start_p - 1, 4),
                "P95_Return": round(p95 / start_p - 1, 4),
                **{k: round(v, 4) for k, v in risk.items()},
            })

            # Full distribution for CSV export
            for idx in range(N_PATHS):
                dist_rows.append({
                    "Scenario": sc["label"],
                    "Asset": asset,
                    "Path": idx,
                    "FinalPrice": round(final[idx], 2),
                    "TotalReturn": round(final[idx] / start_p - 1, 4),
                })

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_CSV, index=False)

    dist = pd.DataFrame(dist_rows)
    dist.to_csv(OUTPUT_DIST, index=False)

    print(f"  Summary saved: {OUTPUT_CSV}")
    print(f"  Distribution saved: {OUTPUT_DIST}  ({len(dist)} rows)")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
