# Updated Feature List for Each Timestamp

## Calendar Features

| Feature | Type | Why it helps |
|---------|------|-------------|
| `hour` | numeric | Captures daily cycle |
| `dayofweek` | numeric | Captures weekly pattern |
| `is_weekend` | binary | Differentiates weekend load |
| `month` | numeric | Month trend |
| `season` | categorical (0-3) | Captures seasonal pattern |

## Data Features

| Feature | Type | Why it helps |
|---------|------|-------------|
| `apartment_01` to `apartment_20` | numeric | Individual apartment loads (20 units, 4 family types) |
| `pv_1kw` | numeric | PV generation from PVGIS |

## Season Encoding

- **0** = Winter (December, January, February)
- **1** = Spring (March, April, May)
- **2** = Summer (June, July, August)
- **3** = Autumn (September, October, November)

