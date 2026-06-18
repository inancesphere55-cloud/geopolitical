import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from statsmodels.tsa.stattools import adfuller

START = "2020-01-01"
END = datetime.today().strftime("%Y-%m-%d")
TICKERS = {
    "Brent Crude": "BZ=F",
    "WTI Crude": "CL=F",
    "Gold": "GC=F",
    "VIX": "^VIX",
}
OUTPUT = "../data/energy_commodity_market_data.csv"
ANN_FACTOR = np.sqrt(252)


def compute_rolling_alpha(beta, ret_asset, ret_bench, window):
    """Rolling conditional alpha = ret_asset - beta * ret_bench over window."""
    return ret_asset.rolling(window).mean() - beta * ret_bench.rolling(window).mean()


def main():
    print(f"[data_pipeline] Downloading {len(TICKERS)} tickers from {START} to {END} ...")
    data = yf.download(
        list(TICKERS.values()),
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
    )

    adj_close = data["Adj Close"].copy()
    adj_close.columns = list(TICKERS.keys())
    adj_close = adj_close.ffill().dropna()
    print(f"  Rows after cleaning: {len(adj_close)}")

    # -- Log returns ----------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        for col in ["Brent Crude", "WTI Crude", "Gold"]:
            tag = col.split()[0]
            adj_close[f"{tag}_LogRet"] = np.log(
                adj_close[col] / adj_close[col].shift(1)
            )

    # -- Volatility features ----------------------------------------
    for asset in ["Brent", "WTI", "Gold"]:
        lr_col = f"{asset}_LogRet"
        adj_close[f"{asset}_RealVol_21d"] = adj_close[lr_col].rolling(21).std() * ANN_FACTOR
        adj_close[f"{asset}_RealVol_63d"] = adj_close[lr_col].rolling(63).std() * ANN_FACTOR

    # -- Spreads ----------------------------------------------------
    adj_close["Brent_WTI_Spread"] = adj_close["Brent Crude"] - adj_close["WTI Crude"]
    adj_close["Brent_Gold_Ratio"] = adj_close["Brent Crude"] / adj_close["Gold"]

    # -- Rolling correlations (63-day window) -----------------------
    adj_close["Corr_BrentGold_63d"] = (
        adj_close["Brent_LogRet"].rolling(63).corr(adj_close["Gold_LogRet"])
    )
    adj_close["Corr_BrentVIX_63d"] = (
        adj_close["Brent_LogRet"].rolling(63).corr(adj_close["VIX"].pct_change())
    )

    # -- Rolling betas (Brent as market proxy) ----------------------
    brent_ret = adj_close["Brent_LogRet"]
    for asset, tag in [("Gold", "Gold"), ("WTI", "WTI")]:
        ret_asset = adj_close[f"{tag}_LogRet"]
        rolling_cov = ret_asset.rolling(63).cov(brent_ret)
        rolling_var = brent_ret.rolling(63).var()
        adj_close[f"Beta_{tag}_63d"] = rolling_cov / rolling_var

    # -- Tail-risk features ----------------------------------------
    adj_close["Brent_LogRet_Skew_21d"] = adj_close["Brent_LogRet"].rolling(21).skew()
    adj_close["Brent_LogRet_Kurt_21d"] = adj_close["Brent_LogRet"].rolling(21).kurt()
    adj_close["Brent_Downside_Vol_21d"] = (
        adj_close["Brent_LogRet"]
        .rolling(21)
        .apply(lambda x: np.sqrt(np.mean(x[x < 0] ** 2)) if (x < 0).sum() > 0 else np.nan)
    )

    final = adj_close.dropna()
    final.to_csv(OUTPUT)
    print(f"  Exported {len(final)} rows x {len(final.columns)} cols to {OUTPUT}")

    # -- Statistical diagnostics -----------------------------------
    lrets = final["Brent_LogRet"]
    jb_stat, jb_p = stats.jarque_bera(lrets)
    adf_stat, adf_p = adfuller(lrets.dropna())[:2]

    print(f"\n  -- Brent Log-Return Diagnostics --")
    print(f"  Observations:       {len(lrets)}")
    print(f"  Mean (daily):       {lrets.mean():.6f}")
    print(f"  Std (daily):        {lrets.std():.6f}")
    print(f"  Skewness:           {lrets.skew():+.4f}")
    print(f"  Kurtosis (excess):  {lrets.kurtosis():+.4f}")
    print(f"  Min / Max:          {lrets.min():+.4f} / {lrets.max():+.4f}")
    print(f"  Jarque-Bera:        {jb_stat:.2f} (p={jb_p:.4f})")
    print(f"  ADF Statistic:      {adf_stat:.4f} (p={adf_p:.4f})")
    print(f"  Ann. Vol (simple):  {lrets.std() * ANN_FACTOR:.2%}")
    print()

    # -- Feature count summary -------------------------------------
    groups = {
        "Prices": 4,
        "Log Returns": 3,
        "Realized Vol": 6,
        "Spreads / Ratios": 2,
        "Rolling Correlations": 2,
        "Rolling Betas": 2,
        "Tail-Risk Stats": 3,
    }
    print(f"  -- Feature Groups --")
    for g, n in groups.items():
        print(f"  {g:<25} {n:>2} features")
    print(f"  {'TOTAL':<25} {sum(groups.values()):>2} features")


if __name__ == "__main__":
    main()
