---not relevant for the analysis---

import pandas as pd
import geopandas as gpd
import shapely

# Paths to your input files
suitability = '/ARE_new_polygon.gpkg'
biogeo_file = '/N2020_Revision_BiogeoRegion.shp'

# Read the input files
try:
    suitability_gdf = gpd.read_file(suitability)
    biogeo_gdf = gpd.read_file(biogeo_file)
except Exception as e:
    print(f"Error reading files: {e}")
    exit()

# Debug input data
print(f"Suitability GDF rows: {len(suitability_gdf)}")
print(f"Biogeo GDF rows: {len(biogeo_gdf)}")
print("Suitability columns:", suitability_gdf.columns.tolist())
print("Suitability geometry types:", suitability_gdf.geometry.geom_type.value_counts().to_dict())
print("Biogeo geometry types:", biogeo_gdf.geometry.geom_type.value_counts().to_dict())
print(f"Initial Suitability CRS: {suitability_gdf.crs}")
print(f"Biogeo CRS: {biogeo_gdf.crs}")

# Check and set source CRS
if suitability_gdf.crs is None or str(suitability_gdf.crs).startswith('LOCAL_CS'):
    print("Warning: Suitability GDF has undefined or invalid CRS.")
    # Try EPSG:4326 (WGS84) as fallback; change to EPSG:21781 or other if known
    source_crs = 'EPSG:4326'  # CHANGE THIS IF YOU KNOW THE CORRECT CRS (e.g., 'EPSG:21781')
    print(f"Overriding CRS to {source_crs}...")
    suitability_gdf.set_crs(source_crs, allow_override=True, inplace=True)

# Validate geometries before reprojection
print("Checking suitability geometries...")
suitability_gdf['is_valid'] = suitability_gdf.geometry.is_valid
suitability_gdf['is_empty'] = suitability_gdf.geometry.is_empty
suitability_gdf['has_coords'] = suitability_gdf.geometry.apply(
    lambda geom: geom is not None and not shapely.is_missing(geom)
)

invalid_geoms = suitability_gdf[~suitability_gdf['is_valid']]
empty_geoms = suitability_gdf[suitability_gdf['is_empty']]
no_coords_geoms = suitability_gdf[~suitability_gdf['has_coords']]

print(f"Invalid geometries in suitability: {len(invalid_geoms)}")
print(f"Empty geometries in suitability: {len(empty_geoms)}")
print(f"Geometries with no coordinates in suitability: {len(no_coords_geoms)}")

# Save problematic geometries
if not invalid_geoms.empty:
    invalid_geoms.to_file('invalid_suitability_geoms.gpkg', driver='GPKG')
    print(f"Saved {len(invalid_geoms)} invalid geometries to 'invalid_suitability_geoms.gpkg'")
if not empty_geoms.empty:
    empty_geoms.to_file('empty_suitability_geoms.gpkg', driver='GPKG')
    print(f"Saved {len(empty_geoms)} empty geometries to 'empty_suitability_geoms.gpkg'")
if not no_coords_geoms.empty:
    no_coords_geoms.to_file('no_coords_suitability_geoms.gpkg', driver='GPKG')
    print(f"Saved {len(no_coords_geoms)} geometries with no coordinates to 'no_coords_suitability_geoms.gpkg'")

# Filter out problematic geometries
original_rows = len(suitability_gdf)
suitability_gdf = suitability_gdf[suitability_gdf['is_valid'] & ~suitability_gdf['is_empty'] & suitability_gdf['has_coords']]
print(f"Filtered out {original_rows - len(suitability_gdf)} invalid/empty geometries. Remaining rows: {len(suitability_gdf)}")

if suitability_gdf.empty:
    print("Error: All suitability geometries are invalid or empty. Check 'ARE_new_polygon.gpkg' in QGIS.")
    exit()

# Reproject to EPSG:2056
try:
    suitability_gdf = suitability_gdf.to_crs(epsg=2056)
    print(f"Suitability CRS after reprojection: {suitability_gdf.crs}")
except Exception as e:
    print(f"Error reprojecting to EPSG:2056: {e}")
    print(f"Failed to reproject from {suitability_gdf.crs} to EPSG:2056.")
    print("Please verify the source CRS of 'ARE_new_polygon.gpkg' (e.g., try EPSG:21781 or check in QGIS).")
    exit()

# Check geometries after reprojection
invalid_suitability = suitability_gdf[~suitability_gdf.geometry.is_valid]
if not invalid_suitability.empty:
    print(f"Invalid geometries in suitability after reprojection: {len(invalid_suitability)}")
    print("Attempting to fix invalid geometries...")
    suitability_gdf['geometry'] = suitability_gdf.geometry.buffer(0)
    invalid_suitability_after = suitability_gdf[~suitability_gdf.geometry.is_valid]
    print(f"Invalid geometries after buffer(0): {len(invalid_suitability_after)}")
    if not invalid_suitability_after.empty:
        print("Applying small positive buffer (0.001m)...")
        suitability_gdf['geometry'] = suitability_gdf.geometry.buffer(0.001)

# Check biogeo geometries
invalid_biogeo = biogeo_gdf[~biogeo_gdf.geometry.is_valid]
print(f"Invalid geometries in biogeo: {len(invalid_biogeo)}")
if not invalid_biogeo.empty:
    print("Fixing invalid biogeo geometries...")
    biogeo_gdf['geometry'] = biogeo_gdf.geometry.buffer(0)

# Check spatial bounds
try:
    print("Suitability bounds:", suitability_gdf.total_bounds)
    print("Biogeo bounds:", biogeo_gdf.total_bounds)
except Exception as e:
    print(f"Error computing bounds: {e}")
    exit()

# Calculate areas (no 'area' column in .gpkg)
print("Calculating areas...")
suitability_gdf['area_ha'] = suitability_gdf.geometry.area / 10000
print(f"Calculated areas. Sample areas (ha): {suitability_gdf['area_ha'].head().to_list()}")

# Debug area issues
print(f"Polygons with NaN areas: {suitability_gdf['area_ha'].isna().sum()}")
print(f"Polygons with zero/near-zero areas (<=0.0001 ha): {(suitability_gdf['area_ha'] <= 0.0001).sum()}")

# Perform spatial join with 'intersects'
print("Performing spatial join with 'intersects'...")
joined_gdf = gpd.sjoin(suitability_gdf, biogeo_gdf, how='left', predicate='intersects')
print(f"Joined GDF rows: {len(joined_gdf)}")
print(f"Rows with NaN in RegionNumm: {joined_gdf['RegionNumm'].isna().sum()}")

# Save unassigned polygons
unassigned_gdf = joined_gdf[joined_gdf['RegionNumm'].isna()]
if not unassigned_gdf.empty:
    unassigned_gdf.to_file('unassigned_polygons.gpkg', driver='GPKG')
    print(f"Saved {len(unassigned_gdf)} unassigned polygons to 'unassigned_polygons.gpkg'")

# Check priority column
priority_col = 'VALUE'
if priority_col not in joined_gdf.columns:
    print(f"Error: Priority column '{priority_col}' not found.")
    print("Available columns:", joined_gdf.columns.tolist())
    exit()

# Group by priority and region
region_col = 'RegionNumm'
summary_df = joined_gdf.groupby([priority_col, region_col])['area_ha'].sum().reset_index()
print(f"Summary DF rows: {len(summary_df)}")

# If empty, diagnose further
if summary_df.empty:
    print("Summary DF is empty. Final diagnostics...")
    print("Sample suitability geometry bounds (first 5):")
    for idx, geom in suitability_gdf.geometry.head(5).items():
        print(f"Polygon {idx}: {geom.bounds if geom is not None else 'None'}")
    print("Sample RegionNumm values in biogeo_gdf:", biogeo_gdf['RegionNumm'].head().to_list())
    print("Consider visualizing 'unassigned_polygons.gpkg' and biogeo shapefile in QGIS.")
    exit()

# Calculate total area per priority
total_area_per_priority = joined_gdf.groupby(priority_col)['area_ha'].sum().reset_index()
total_area_per_priority = total_area_per_priority.rename(columns={'area_ha': 'total_area_ha'})
print(f"Total area per priority (ha):")
print(total_area_per_priority)

# Merge and calculate percentages
summary_df = summary_df.merge(total_area_per_priority, on=priority_col)
summary_df['percentage'] = (summary_df['area_ha'] / summary_df['total_area_ha'] * 100).round(2)

# Format for CSV
summary_df['area_ha'] = summary_df['area_ha'].round(2)
summary_df = summary_df[[priority_col, region_col, 'area_ha', 'percentage']].rename(
    columns={
        priority_col: 'Priority',
        region_col: 'Region',
        'area_ha': 'Area_ha',
        'percentage': 'Percentage'
    }
)

# Save to CSV
output_csv = 'priority_by_biogeo_percentage_table.csv'
summary_df.to_csv(output_csv, index=False)
print(f"CSV file '{output_csv}' has been saved.")
print("First few rows of summary:")
print(summary_df.head(10))

# Pivot table
pivot_df = summary_df.pivot_table(
    index='Region',
    columns='Priority',
    values='Percentage',
    fill_value=0
).round(2)
print("\nPivot table of percentages (Priority vs Region):")
print(pivot_df)
