"""
Interactive viewer for PV forecasting results.
Opens all available result figures.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread

results_dir = Path(__file__).parent / "results"

print("Opening result figures...")
print()

figures = list(results_dir.glob("*.png"))

if not figures:
    print("No result figures found in results/ directory")
    sys.exit(1)

for fig_path in figures:
    print(f"Displaying: {fig_path.name}")
    img = imread(fig_path)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(fig_path.stem.replace('_', ' ').title(), fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show(block=False)

print()
print(f"Displayed {len(figures)} figure(s). Close the windows when done.")
input("Press Enter to close all figures...")
plt.close('all')

