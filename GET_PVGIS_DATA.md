# Get PVGIS Data for Buildings

## Quick Start

### Option 1: Use Existing Torino PVGIS File (Recommended if all buildings are close)

If all your buildings are in Torino with similar coordinates, you can use one PVGIS file for all:

1. **Update the script** (`query_pvgis_for_buildings.py`):
   ```python
   use_existing_torino = True
   torino_pvgis_path = 'data/pvgis_torino_hourly.csv'  # Your PVGIS file path
   ```

2. **Run the script**:
   ```bash
   python query_pvgis_for_buildings.py
   ```

3. **Result**: Each building gets the same PVGIS data (saved to `data/pvgis_buildings/`)

### Option 2: Query PVGIS API for Each Building

If buildings have significantly different coordinates:

1. **Update the script**:
   ```python
   use_existing_torino = False
   ```

2. **Run the script**:
   ```bash
   python query_pvgis_for_buildings.py
   ```

3. **Note**: PVGIS API has rate limits - the script will take time for many buildings

## Installation

Make sure you have the `requests` library:

```bash
pip install requests
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Building Data Format

Your CSV should have:
- `OSM ID` column (or any building identifier)
- `Latitude` column
- `Longitude` column

The script will automatically detect these columns.

## Output

PVGIS data files will be saved to `data/pvgis_buildings/`:
- `pvgis_building_1.csv`
- `pvgis_building_2.csv`
- etc.

Each file contains:
- `time` column (datetime index)
- `pv_power` column (in kW)

## PVGIS API Parameters

The script uses default PVGIS parameters:
- PV Power: 1 kWp
- System Loss: 14%
- Slope: 30°
- Azimuth: 180° (south-facing)
- Technology: Crystalline Silicon

You can modify these in the `query_pvgis_api()` function if needed.


