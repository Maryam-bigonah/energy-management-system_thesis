from __future__ import annotations

from typing import Any, Dict


def fetch_pvgis_metadata(latitude: float, longitude: float) -> Dict[str, Any]:
    """Stub: fetch PVGIS metadata for a location.

    Replace with real API calls using pvlib or requests.
    """
    return {"lat": latitude, "lon": longitude, "source": "pvgis"}
