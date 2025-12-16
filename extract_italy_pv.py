import pandas as pd

df = pd.read_csv("time_series.csv", parse_dates=['utc_timestamp'])

# Select only Italy solar PV generation
italy_pv = df[['utc_timestamp', 'generation_solar_it']]

italy_pv.to_csv("italy_pv_hourly.csv", index=False)
