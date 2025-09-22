How to use Load Profile Generator (LPG) CSVs here

1) In LPG, export CSVs for each archetype with 1-minute resolution and include:
   - Timestamp column
   - Total electricity (Wh)
   - By-appliance columns if available (Wh)

2) Save CSVs with these filenames in this folder:
   - couple_working.csv
   - family_one_child.csv
   - one_works_one_home.csv
   - retired_couple.csv

3) Expected columns (case-insensitive; flexible names):
   - timestamp (or time)
   - total (or total_electricity/total electricity)
   - appliance columns like Fridge, Dishwasher, HVAC, Lighting, etc.

4) After placing files, run:

```bash
python scripts/run_lpg_pipeline.py
```

Validation will fail if totals and category sums differ more than ±0.5% daily, or if negative values exist.
