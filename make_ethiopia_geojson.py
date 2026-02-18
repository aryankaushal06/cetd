from pathlib import Path
import geopandas as gpd

OUT = Path("data/ethiopia.geojson")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Use Natural Earth via GeoPandas built-in download URL (not geodatasets)
URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

world = gpd.read_file(URL).to_crs("EPSG:4326")

# Natural Earth uses ADMIN for country name in this file
eth = world[world["ADMIN"] == "Ethiopia"]
if eth.empty:
    raise RuntimeError("Ethiopia not found in Natural Earth countries dataset.")

eth = eth.dissolve()
eth.to_file(OUT, driver="GeoJSON")
print("Wrote", OUT)
