import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich.box import MINIMAL, HEAVY, ROUNDED
from rich.align import Align

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    ("data_pipeline.py", "../data/energy_commodity_market_data.csv"),
    ("regime_model.py", "../data/market_regime_output.csv"),
    ("scenario_simulator.py", "../data/multi_asset_stress_matrix.csv"),
    ("generate_impact_sheet.py", "../outputs/Energy_Commodity_Shock_Model.xlsx"),
    ("generate_report.py", "../outputs/Energy_Commodity_Geopolitical_Risk_Report.pdf"),
]

BASE_BRENT = 79.55
BASE_GOLD = 4358.90
ANN_FACTOR = np.sqrt(252)

console = Console()


def run_pipeline():
    total = len(SCRIPTS)
    for i, (script, _) in enumerate(SCRIPTS, 1):
        script_path = ROOT / script
        console.print(f"[bold cyan][{i}/{total}][/] Running {script} ...")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            console.print(f"  [red]FAILED[/] (exit {result.returncode})")
            console.print(result.stderr)
            sys.exit(1)
        for line in result.stdout.strip().split("\n"):
            console.print(f"  {line}")
        console.print(f"  [green]Completed in {elapsed:.1f}s[/]")


def load_data():
    data_dir = ROOT / "../data"
    outputs_dir = ROOT / "../outputs"

    market = pd.read_csv(data_dir / "energy_commodity_market_data.csv", index_col="Date", parse_dates=True)
    regime = pd.read_csv(data_dir / "market_regime_output.csv", index_col="Date", parse_dates=True)
    stress = pd.read_csv(data_dir / "multi_asset_stress_matrix.csv")

    last_prices = {}
    for col in ["Brent Crude", "WTI Crude", "Gold", "VIX"]:
        last_prices[col] = market[col].iloc[-1]

    brent_lr = market["Brent_LogRet"].dropna()
    brent_vol = brent_lr.std() * ANN_FACTOR

    crisis_prob = regime["High_Vol_Probability"].dropna().iloc[-1] if "High_Vol_Probability" in regime.columns else None

    scenarios = {}
    for _, row in stress.iterrows():
        key = (row["Scenario"], row["Asset"])
        scenarios[key] = row

    corr_normal = None
    corr_crisis = None
    for f in data_dir.glob("regime_correlations_*.csv"):
        if "normal" in f.stem:
            corr_normal = pd.read_csv(f, index_col=0)
        elif "crisis" in f.stem:
            corr_crisis = pd.read_csv(f, index_col=0)

    return {
        "market": market,
        "regime": regime,
        "stress": stress,
        "last_prices": last_prices,
        "brent_vol": brent_vol,
        "crisis_prob": crisis_prob,
        "scenarios": scenarios,
        "corr_normal": corr_normal,
        "corr_crisis": corr_crisis,
    }


def make_header():
    table = Table.grid(padding=0)
    table.add_column()
    title = Text()
    title.append("SHESHA FINTECH & AI", style="bold bright_cyan")
    title.append("\n")
    title.append("ENERGY & COMMODITY SHOCK ANALYSIS", style="bold white")
    title.append("\n")
    title.append(f"Live Terminal Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim white")
    table.add_row(Align.center(title))
    return Panel(table, box=HEAVY, style="bright_cyan")


def make_market_panel(data):
    prices = data["last_prices"]
    table = Table(box=MINIMAL, show_header=False, padding=(0, 2))
    table.add_column("Asset", style="bold white")
    table.add_column("Price", justify="right")

    emojis = {"Brent Crude": "Brent", "WTI Crude": "WTI", "Gold": "Gold", "VIX": "VIX"}

    for name, label in emojis.items():
        val = prices[name]
        if name == "VIX":
            table.add_row(label, f"{val:.2f}")
        else:
            table.add_row(label, f"${val:.2f}")

    table.add_section()
    table.add_row("[dim]Ann. Vol (Brent)[/]", f"[yellow]{data['brent_vol']:.2%}[/]")

    return Panel(table, title="[bold]MARKET PRICES[/]", box=ROUNDED, border_style="cyan")


def make_regime_panel(data):
    prob = data["crisis_prob"]
    if prob is None:
        return Panel("[red]No regime data[/]", title="[bold]REGIME STATE[/]", box=ROUNDED, border_style="red")

    gauge_chars = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))

    color = "green" if prob < 0.3 else ("yellow" if prob < 0.6 else "red")

    status = "NORMAL" if prob < 0.3 else ("ELEVATED" if prob < 0.6 else "CRISIS")

    table = Table.grid(padding=(0, 1))
    table.add_row("[bold]Crisis Probability[/]", f"[{color}]{prob:.2%}[/]")
    table.add_row("[bold]Gauge[/]", f"[{color}]{gauge_chars}[/]")
    table.add_row("[bold]Status[/]", f"[bold {color}]{status}[/]")

    return Panel(table, title="[bold]REGIME STATE[/]", box=ROUNDED, border_style=color)


def make_correlation_panel(data):
    if data["corr_normal"] is None or data["corr_crisis"] is None:
        return Panel("[red]No correlation data[/]", title="[bold]CORRELATIONS[/]", box=ROUNDED, border_style="red")

    t = Table(box=MINIMAL, show_header=False, padding=(0, 2))
    t.add_column("Regime", style="bold")
    t.add_column("Brent-Gold", justify="right")
    t.add_column("Brent-VIX", justify="right")

    try:
        brent_gold_normal = data["corr_normal"].loc["Brent_LogRet", "Gold_LogRet"]
        brent_vix_normal = data["corr_normal"].loc["Brent_LogRet", "VIX_Ret"]
    except Exception:
        brent_gold_normal = brent_vix_normal = 0

    try:
        brent_gold_crisis = data["corr_crisis"].loc["Brent_LogRet", "Gold_LogRet"]
        brent_vix_crisis = data["corr_crisis"].loc["Brent_LogRet", "VIX_Ret"]
    except Exception:
        brent_gold_crisis = brent_vix_crisis = 0

    t.add_row("[green]Normal[/]", f"{brent_gold_normal:+.2f}", f"{brent_vix_normal:+.2f}")
    t.add_row("[red]Crisis[/]", f"{brent_gold_crisis:+.2f}", f"{brent_vix_crisis:+.2f}")

    return Panel(t, title="[bold]REGIME CORRELATIONS[/]", box=ROUNDED, border_style="magenta")


def make_var_panel(data):
    scenarios_data = data["scenarios"]
    t = Table(box=MINIMAL, show_edge=False)
    t.add_column("Scenario", style="bold")
    t.add_column("Asset")
    t.add_column("VaR 95%", justify="right")
    t.add_column("CVaR 95%", justify="right")
    t.add_column("VaR 99%", justify="right")
    t.add_column("Max DD", justify="right")

    for sc_key, label in [("A", "Status Quo"), ("B", "Moderate"), ("C", "Severe")]:
        for asset in ["Brent", "Gold"]:
            key = (f"{sc_key}: {label} (State 0)" if sc_key == "A"
                   else f"{sc_key}: {label} Escalation" if sc_key == "B"
                   else f"{sc_key}: {label} Disruption", asset)
            if key in scenarios_data:
                r = scenarios_data[key]
                sc_label = label if asset == "Brent" else ""
                t.add_row(
                    sc_label,
                    asset,
                    f"{r['VaR_95%']:.1%}",
                    f"{r['CVaR_95%']:.1%}",
                    f"{r['VaR_99%']:.1%}",
                    f"{r['Max_Drawdown']:.1%}",
                )

    return Panel(t, title="[bold]RISK METRICS (VaR / CVaR)[/]", box=ROUNDED, border_style="red")


def make_scenario_panel(data):
    scenarios_data = data["scenarios"]
    t = Table(box=MINIMAL, show_edge=False)
    t.add_column("Scenario", style="bold")
    t.add_column("Brent ($)", justify="right")
    t.add_column("Gold ($)", justify="right")
    t.add_column("Brent Ret", justify="right")
    t.add_column("Gold Ret", justify="right")

    for sc_key, label in [("A", "Status Quo"), ("B", "Mod."), ("C", "Sev.")]:
        brent_key = (f"{sc_key}: {label} (State 0)" if sc_key == "A"
                     else f"{sc_key}: {label} Escalation" if sc_key == "B"
                     else f"{sc_key}: {label} Disruption", "Brent")
        gold_key = (f"{sc_key}: {label} (State 0)" if sc_key == "A"
                    else f"{sc_key}: {label} Escalation" if sc_key == "B"
                    else f"{sc_key}: {label} Disruption", "Gold")

        if brent_key in scenarios_data and gold_key in scenarios_data:
            b = scenarios_data[brent_key]
            g = scenarios_data[gold_key]
            color = "green" if sc_key == "A" else ("yellow" if sc_key == "B" else "red")
            t.add_row(
                f"[{color}]{label}[/]",
                f"${b['Median_Price']:.2f}",
                f"${g['Median_Price']:.2f}",
                f"{b['Median_Return']:+.1%}",
                f"{g['Median_Return']:+.1%}",
            )

    return Panel(t, title="[bold]SCENARIO OUTLOOK (30D)[/]", box=ROUNDED, border_style="yellow")


def make_footer():
    pdf_path = ROOT / "../outputs/Energy_Commodity_Geopolitical_Risk_Report.pdf"
    pdf_status = f"[green]PDF Report Ready[/]" if pdf_path.exists() else "[dim]PDF report pending[/]"
    help_text = f"[dim]Ctrl+C to exit  |  {pdf_status}  |  --refresh for auto-update every 5min[/]"
    return Panel(Align.center(help_text), box=HEAVY, style="bright_cyan")


def build_dashboard(data):
    layout = Layout()
    layout.split_column(
        Layout(make_header(), size=5),
        Layout(name="main"),
        Layout(make_footer(), size=3),
    )

    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    layout["left"].split_column(
        Layout(make_market_panel(data)),
        Layout(make_regime_panel(data)),
        Layout(make_correlation_panel(data)),
    )

    layout["right"].split_column(
        Layout(make_var_panel(data)),
        Layout(make_scenario_panel(data)),
    )

    return layout


def main():
    console.clear()
    console.print("[bold green]Running full analysis pipeline...[/]")
    run_pipeline()
    console.print("[bold green]Loading results...[/]")
    data = load_data()

    refresh_mode = "--refresh" in sys.argv

    with Live(build_dashboard(data), refresh_per_second=1, screen=False) as live:
        try:
            if refresh_mode:
                while True:
                    time.sleep(300)
                    console.print("[dim]Refreshing pipeline...[/]")
                    run_pipeline()
                    data = load_data()
                    live.update(build_dashboard(data))
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Dashboard closed.[/]")


if __name__ == "__main__":
    main()
