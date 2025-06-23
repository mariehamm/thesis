import geopandas as gpd
from pyproj import CRS


'''reprojecting new ARE data to epgs2056'''
# Path to your GeoPackage
# test
file_path = "/home/ubuntu/master/data/ARE_new_polygon.gpkg"

# Load the GeoPackage
gdf = gpd.read_file(file_path)

# Display basic information
print("Dataset Info:")
print(gdf.info())
print("\nFirst few rows:")
print(gdf.head())

# Check the current CRS
print("\nOriginal CRS:", gdf.crs)

# Check sample coordinates to verify spatial context
print("\nSample coordinates (first geometry):")
print(gdf.geometry.iloc[0])

# Define EPSG:2056 CRS explicitly
target_crs = CRS.from_epsg(2056)

# Force-assign the correct CRS if the geometry values match EPSG:2056 coordinates
print("Assigning EPSG:2056 (Swiss CH1903+ / LV95) as the CRS, overriding the current one.")
gdf.set_crs(target_crs, inplace=True, allow_override=True)

# Define EPSG:2056 CRS explicitly
target_crs = CRS.from_epsg(2056)

# Force assign the CRS regardless of existing LOCAL_CS
print("Assigning EPSG:2056 (Swiss CH1903+ / LV95) as the CRS, overriding the current one.")
gdf.set_crs(target_crs, inplace=True, allow_override=True)


# Verify the new CRS
print("\nUpdated CRS:", gdf.crs)
print("Updated CRS (EPSG code):", gdf.crs.to_epsg())

# Verify coordinates after reprojection
print("\nSample coordinates after update (first geometry):")
print(gdf.geometry.iloc[0])

# Save the updated GeoPackage
output_path = "/home/ubuntu/master/data/ARE_new_polygon_epsg2056.gpkg"
gdf.to_file(output_path, driver="GPKG")
print(f"GeoPackage saved with EPSG:2056 to: {output_path}")

gdf = gpd.read_file("/home/ubuntu/master/data/ARE_new_polygon_epsg2056.gpkg")
print("CRS:", gdf.crs)
print("EPSG code:", gdf.crs.to_epsg())
