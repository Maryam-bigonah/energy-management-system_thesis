# Feature List - Detailed Summary

| Feature | Type | Units | Family Type | Why it helps |
|---------|------|-------|--------------|--------------|
| `apartment_01` | numeric | kWh/hour (≈kW) | couple_working | Load for apartment 1 (couple_working) |
| `apartment_02` | numeric | kWh/hour (≈kW) | couple_working | Load for apartment 2 (couple_working) |
| `apartment_03` | numeric | kWh/hour (≈kW) | couple_working | Load for apartment 3 (couple_working) |
| `apartment_04` | numeric | kWh/hour (≈kW) | couple_working | Load for apartment 4 (couple_working) |
| `apartment_05` | numeric | kWh/hour (≈kW) | couple_working | Load for apartment 5 (couple_working) |
| `apartment_06` | numeric | kWh/hour (≈kW) | family_one_child | Load for apartment 6 (family_one_child) |
| `apartment_07` | numeric | kWh/hour (≈kW) | family_one_child | Load for apartment 7 (family_one_child) |
| `apartment_08` | numeric | kWh/hour (≈kW) | family_one_child | Load for apartment 8 (family_one_child) |
| `apartment_09` | numeric | kWh/hour (≈kW) | family_one_child | Load for apartment 9 (family_one_child) |
| `apartment_10` | numeric | kWh/hour (≈kW) | family_one_child | Load for apartment 10 (family_one_child) |
| `apartment_11` | numeric | kWh/hour (≈kW) | one_working | Load for apartment 11 (one_working) |
| `apartment_12` | numeric | kWh/hour (≈kW) | one_working | Load for apartment 12 (one_working) |
| `apartment_13` | numeric | kWh/hour (≈kW) | one_working | Load for apartment 13 (one_working) |
| `apartment_14` | numeric | kWh/hour (≈kW) | one_working | Load for apartment 14 (one_working) |
| `apartment_15` | numeric | kWh/hour (≈kW) | one_working | Load for apartment 15 (one_working) |
| `apartment_16` | numeric | kWh/hour (≈kW) | retired | Load for apartment 16 (retired) |
| `apartment_17` | numeric | kWh/hour (≈kW) | retired | Load for apartment 17 (retired) |
| `apartment_18` | numeric | kWh/hour (≈kW) | retired | Load for apartment 18 (retired) |
| `apartment_19` | numeric | kWh/hour (≈kW) | retired | Load for apartment 19 (retired) |
| `apartment_20` | numeric | kWh/hour (≈kW) | retired | Load for apartment 20 (retired) |
| `pv_1kw` | numeric | kW | N/A | PV generation from PVGIS |
| `hour` | numeric | 0-23 | N/A | Captures daily cycle |
| `dayofweek` | numeric | 0-6 (Mon=0, Sun=6) | N/A | Captures weekly pattern |
| `month` | numeric | 1-12 | N/A | Month trend |
| `is_weekend` | binary | 0 or 1 | N/A | Differentiates weekend load |
| `season` | categorical | 0-3 (winter, spring, summer, autumn) | N/A | Captures seasonal pattern |


## Season Mapping

| Season Code | Season Name | Months |
|-------------|-------------|--------|
| 0 | Winter | December, January, February |
| 1 | Spring | March, April, May |
| 2 | Summer | June, July, August |
| 3 | Autumn | September, October, November |
