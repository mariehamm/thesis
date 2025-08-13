'''This Python script analyzes species richness across Switzerland by mapping occurrence data of mammals, amphibians, reptiles, birds 
onto a hexagonal grid in the LV95 (EPSG:2056) coordinate system.
It combines species datasets, applies jittering to separate overlapping points, 
performs a spatial join with a pre-generated hexagonal grid, calculates species richness 
per hexagon, and computes area-weighted richness and density for border hexagons. 
The script also tests the normality of spatial coordinates and
species richness.'''

#Import libraries
import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from scipy.stats import normaltest
import vertebrates
import geometries

# Access the DataFrames from vertebrates & geometries module
mammals = vertebrates.mammals
amphibians = vertebrates.amphibians
reptiles = vertebrates.reptiles
birds = vertebrates.birds

hex_grid_gdf= geometries.hex_grid_gdf
swiss_boundary = geometries.swiss_boundary


'''Preparation for species richness'''

# Define columns to keep from each vertebrate dataset
columns = ['class', 'species', 'decimalLatitude', 'decimalLongitude', 'year']

# Combine vertebrate datasets into a single DataFrame
combined_data = pd.concat([
    df[columns] for df in [mammals, amphibians, reptiles, birds]
], ignore_index=True)


# Convert to GeoDataFrame with WGS84 (EPSG:4326)
gdf_species = gpd.GeoDataFrame(
    combined_data,
    geometry=[Point(lon, lat) for lon, lat in zip(combined_data['decimalLongitude'], combined_data['decimalLatitude'])],
    crs="EPSG:4326"
)
# Reproject to Swiss LV95 (EPSG:2056) for consistency with hexagonal grid
gdf_species = gdf_species.to_crs(epsg=2056)


# Define function to jitter point geometries (small random noise)
def jitter_geometry(geometry, amount=0.5):
    """Add small random noise to each point's coordinates to separate identical points."""
    jitter_x = np.random.uniform(-amount, amount, len(geometry))
    jitter_y = np.random.uniform(-amount, amount, len(geometry))
    
    return gpd.GeoSeries(
        [Point(point.x + dx, point.y + dy) for point, dx, dy in zip(geometry, jitter_x, jitter_y)],
        crs=geometry.crs
    )

# Apply jittering to species points to avoid overlap (1 meter max shift)
gdf_species['geometry'] = jitter_geometry(gdf_species['geometry'], amount=1)  


# Perform spatial join to assign each species point to a hexagon
# 'within' ensures points are assigned to the hexagon they fall inside
# 'left' keeps all species points, even if outside hexagons (hex_id will be NaN)
gdf_joined = gpd.sjoin(gdf_species, hex_grid_gdf, how='left', predicate='within')



'''Species Richness calculation'''

# Calculate species richness: number of different species per hexagon
species_richness = gdf_joined.groupby('hex_id')['species'].nunique().reset_index()
species_richness.rename(columns={'species': 'species_richness'}, inplace=True)

# list unique species names per hexagon
species_names = gdf_joined.groupby('hex_id')['species'].apply(lambda x: ', '.join(sorted(set(x.dropna())))).reset_index()
species_names.rename(columns={'species': 'species_list'}, inplace=True)

# list unique class names per hexagon
class_names = gdf_joined.groupby('hex_id')['class'].apply(lambda x: ', '.join(sorted(set(x.dropna())))).reset_index()
class_names.rename(columns={'class': 'class_list'}, inplace=True)

# list unique years per hexagon
year = gdf_joined.groupby('hex_id')['year'].apply(lambda x: ', '.join(sorted(set(x.dropna().astype(str))))).reset_index()


# Merge species richness, species list, class list, and years into the hexagonal grid
hex_grid_species_richness = hex_grid_gdf.merge(species_richness, on='hex_id', how='left')
hex_grid_species_richness = hex_grid_species_richness.merge(species_names, on='hex_id', how='left')
hex_grid_species_richness = hex_grid_species_richness.merge(class_names, on='hex_id', how='left')
hex_grid_species_richness = hex_grid_species_richness.merge(year, on='hex_id', how = 'left' )

# Handle missing values (hexagons with no species points)
hex_grid_species_richness['species_richness'] = hex_grid_species_richness['species_richness'].fillna(0)
hex_grid_species_richness['species_list'] = hex_grid_species_richness['species_list'].fillna('')
hex_grid_species_richness['class_list'] = hex_grid_species_richness['class_list'].fillna('')
hex_grid_species_richness['year'] = hex_grid_species_richness['year'].fillna('')




# Ensure CRS alignment between Swiss boundary and hexagonal grid
if swiss_boundary.crs != hex_grid_gdf.crs:
    swiss_boundary = swiss_boundary.to_crs(hex_grid_gdf.crs)

# Calculate areas for area-weighted richness
hex_grid_gdf['area_total'] = hex_grid_gdf.geometry.area  # Total hexagon area
hex_grid_gdf['area_within_swiss'] = hex_grid_gdf.intersection(swiss_boundary.unary_union).area  # Area inside Switzerland
hex_grid_gdf['area_share'] = hex_grid_gdf['area_within_swiss'] / hex_grid_gdf['area_total']

# Validate area_share (should be between 0 and 1)
hex_grid_gdf['area_share'] = hex_grid_gdf['area_share'].clip(0, 1)  # Ensure no invalid values

# Merge species richness with hex grid
hex_grid_species_richness = hex_grid_gdf.merge(species_richness, on='hex_id', how='left')

# Ensure numeric types for calculation
hex_grid_species_richness['species_richness'] = pd.to_numeric(
    hex_grid_species_richness['species_richness'], errors='coerce'
).fillna(0)
hex_grid_species_richness['area_share'] = pd.to_numeric(
    hex_grid_species_richness['area_share'], errors='coerce'
).fillna(0)


# Calculate weighted species richness
# Weighted richness based on area proportion inside Switzerland
hex_grid_species_richness['species_richness_weighted'] = (
    hex_grid_species_richness['species_richness'] * hex_grid_species_richness['area_share']
)

# Calculate richness density (species per unit area within Switzerland)
hex_grid_species_richness['richness_density'] = (
    hex_grid_species_richness['species_richness'] / hex_grid_species_richness['area_within_swiss'] #density metric (e.g., species per square kilometer). Border hexagons with smaller areas won’t automatically appear “less rich” unless their species density is genuinely lower.
)

# Convert area_within_swiss from m² to km²
hex_grid_species_richness['area_within_swiss_km2'] = hex_grid_species_richness['area_within_swiss'] / 1_000_000

# Calculate richness density (species per km²)
hex_grid_species_richness['richness_density_km2'] = (
    hex_grid_species_richness['species_richness'] / hex_grid_species_richness['area_within_swiss_km2']
).replace([float('inf'), -float('inf')], 0).fillna(0)



# Save the adjusted dataset
hex_grid_species_richness.to_file("/home/ubuntu/master/output/update/spec_rich_7500.geojson", driver='GeoJSON')

# Save to GeoJSON for QGIS



'''Statistical Testing'''

# Perform normality tests on spatial coordinates and species richness
x_coords = gdf_species.geometry.x
y_coords = gdf_species.geometry.y

# Perform D’Agostino’s K-squared Test on coordinates
stat_x, p_x = normaltest(x_coords)
stat_y, p_y = normaltest(y_coords)

# Extract richness values for normality test
richness_values = hex_grid_species_richness['species_richness'].dropna()

# Print coordinate test results
print("Normality Test for Spatial Coordinates (D'Agostino’s K-squared):")
print(f"  Longitude (X): statistic = {stat_x:.3f}, p-value = {p_x:.3f}")
print(f"  Latitude (Y): statistic = {stat_y:.3f}, p-value = {p_y:.3f}")

print("Richness Values Summary:")
print(f"Length: {len(richness_values)}")
print(f"Unique Values: {richness_values.unique()}")
print(f"Non-zero Count: {(richness_values > 0).sum()}")
stat_richness, p_richness = normaltest(richness_values)

print("\nNormality Test for Species Richness (D'Agostino’s K-squared):")
print(f"  Species Richness: statistic = {stat_richness:.3f}, p-value = {p_richness:.3f}")

# Interpret normality test results (alpha = 0.05)
alpha = 0.05
print("\nInterpretation (alpha = 0.05):")
if p_x > alpha and p_y > alpha:
    print("  Spatial coordinates: Likely normal distribution.")
else:
    print("  Spatial coordinates: NOT normally distributed.")
if p_richness > alpha:
    print("  Species richness: Likely normal distribution.")
else:
    print("  Species richness: NOT normally distributed.")

'''
Normality Test for Spatial Coordinates (D'Agostino’s K-squared):
  Longitude (X): statistic = 68578.337, p-value = 0.000
  Latitude (Y): statistic = 113773.258, p-value = 0.000
Richness Values Summary:
Length: 355
Unique Values: [  0. 183. 170. 169. 177. 129.  80. 185. 121. 181. 186. 142.   5. 166.
 180. 149.  10.  86. 194. 212. 151. 123. 136. 156. 213. 175. 137.  45.
 159. 193. 209. 148. 161.   2. 135. 176. 199. 210. 162. 164. 191. 211.
 155. 134. 153.  78. 147. 200. 158. 187. 171. 172. 163. 214. 119. 196.
 150. 152. 178. 189. 104. 126. 165. 146. 188. 127. 125. 203. 174. 160.
 184. 124. 122. 195. 168. 140. 154.   7.  93. 145. 111. 182. 141. 144.
  30.  96. 132. 143. 190. 112.  95. 107. 101. 157. 173. 179.   1. 139.
 208. 167. 128. 114. 116. 197. 120.   3. 110. 138. 117. 201.  83. 130.
  65.  47.   4.  99.  27. 105.   6.  64.  67. 109.  53.  61.  98.  68.
  90. 131.]
Non-zero Count: 337

Normality Test for Species Richness (D'Agostino’s K-squared):
  Species Richness: statistic = 89.839, p-value = 0.000

Interpretation (alpha = 0.05):
  Spatial coordinates: NOT normally distributed.
  Species richness: NOT normally distributed.
'''




