import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    ("data_pipeline.py", "../data/energy_commodity_market_data.csv"),
    ("regime_model.py", "../data/market_regime_output.csv"),
    ("scenario_simulator.py", "../data/multi_asset_stress_matrix.csv"),
    ("generate_impact_sheet.py", "../outputs/Energy_Commodity_Shock_Model.xlsx"),
    ("generate_report.py", "../outputs/Energy_Commodity_Geopolitical_Risk_Report.pdf"),
    ("ai_insights.py", "../data/ai_insights.json"),
]

def main():
    total = len(SCRIPTS)
    root = Path(__file__).resolve().parent
    print("=" * 60)
    print("ENERGY & COMMODITY SHOCK ANALYSIS - FULL PIPELINE")
    print("=" * 60)

    for i, (script, output_rel) in enumerate(SCRIPTS, 1):
        script_path = root / script
        print(f"\n[{i}/{total}] Running {script} ...")
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(root),
            capture_output=False,
            text=True,
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            sys.exit(1)

        print(f"  Completed in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("ALL SCRIPTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated files:")
    for _, output_rel in SCRIPTS:
        p = (root / output_rel).resolve()
        if p.exists():
            print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  {p}  (missing!)")

if __name__ == "__main__":
    main()
