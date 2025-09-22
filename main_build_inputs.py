from __future__ import annotations

from torino_energy.data_sources import gme_gse, lpg, osm_roofs, pvgis, tariffs
from torino_energy.transform import agg_validate, time_align


def main() -> None:
    print("torino_energy scaffold is ready.")
    # Demo calls to ensure imports work
    print(pvgis.fetch_pvgis_metadata(45.07, 7.69))
    print(lpg.fetch_lpg_prices())
    print(tariffs.list_energy_tariffs())
    print(gme_gse.fetch_gme_gse_market_data())
    print(osm_roofs.query_osm_roofs({"type": "FeatureCollection", "features": []}))
    print(time_align.align_time_series(["2024-01-01T00:00:00Z"]))
    print(agg_validate.aggregate([{"k": 1}, {"k": 1}, {"k": 2}], "k"))


if __name__ == "__main__":
    main()
