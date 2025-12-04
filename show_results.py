"""
Display all available forecasting result figures and metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

results_dir = Path(__file__).parent / "results"

print("=" * 80)
print("PV FORECASTING RESULTS SUMMARY")
print("=" * 80)
print()

# Check available results
available_figures = list(results_dir.glob("*.png"))
available_metrics = list(results_dir.glob("*.csv"))

print("Available Result Files:")
print("-" * 80)
for fig in available_figures:
    print(f"  📊 {fig.name} ({fig.stat().st_size / 1024:.1f} KB)")

for csv in available_metrics:
    print(f"  📄 {csv.name}")
    if "metrics" in csv.name:
        df = pd.read_csv(csv)
        print(f"     Contents:")
        print(df.to_string(index=False))
        print()

print()
print("=" * 80)
print("FIGURE DESCRIPTIONS")
print("=" * 80)
print()

if (results_dir / "gradientboosting_results.png").exists():
    print("📊 gradientboosting_results.png")
    print("   This figure contains:")
    print("   - Top panel: 24-hour PV forecast time series")
    print("   - Bottom panel: Performance metrics bar chart")
    print("     * Validation MAE, Validation RMSE")
    print("     * Test MAE, Test RMSE")
    print()
    print("   Metrics shown:")
    metrics_df = pd.read_csv(results_dir / "gradientboosting_metrics.csv")
    for _, row in metrics_df.iterrows():
        print(f"   {row['Set']} Set:")
        print(f"     MAE:  {row['MAE (kW)']:.4f} kW")
        print(f"     RMSE: {row['RMSE (kW)']:.4f} kW")
        print(f"     nRMSE: {row['nRMSE']:.4f}")
        print(f"     R²:   {row['R²']:.4f}")
    print()

print("=" * 80)
print("TO VIEW FIGURES:")
print("=" * 80)
print()
print("Open the following files in an image viewer:")
for fig in available_figures:
    print(f"  {fig.absolute()}")
print()
print("Or run this Python code to display:")
print("  import matplotlib.pyplot as plt")
print("  from pathlib import Path")
for fig in available_figures:
    print(f"  img = plt.imread('{fig}')")
    print(f"  plt.figure(figsize=(12, 8))")
    print(f"  plt.imshow(img)")
    print(f"  plt.axis('off')")
    print(f"  plt.title('{fig.stem}')")
    print(f"  plt.show()")

