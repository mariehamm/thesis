'''This Python script calculates biodiversity indices (Shannon index) for species occurrences mapped onto
a pre-generated hexagonal grid over Switzerland in the LV95 (EPSG:2056) coordinate system. It computes the
Shannon Index per hexagon, along with area-weighted
versions and density metrics (per km²). The script also compares species composition for two specific hexagons
(hex_id 166 and 167) with a visualization. '''

#import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import entropy
from shapely.geometry import Point
import matplotlib.pyplot as plt

import geometries
import richness

# access DataFrames from custom modules
hex_grid_gdf = geometries.hex_grid_gdf
swiss_boundary = geometries.swiss_boundary
swiss_border_gdf = richness.swiss_border_gdf
combined_data = richness.combined_data
gdf_species = richness.gdf_species
gdf_joined = richness.gdf_joined
species_richness = richness.species_richness

# Ensure CRS consistency (all GeoDataFrames in EPSG:2056)
if swiss_border_gdf.crs != "EPSG:2056":
    swiss_border_gdf = swiss_border_gdf.to_crs(epsg=2056)
if hex_grid_gdf.crs != "EPSG:2056":
    hex_grid_gdf = hex_grid_gdf.to_crs(epsg=2056)


'''Shannon Index calculation'''

# https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.stats.entropy.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html


def compute_shannon_index(species_list, base=np.e):
    """Compute Shannon Index with customizable base (default: natural log)."""
    if len(species_list) <= 1 or species_list.isna().all():
        return 0
    species_counts = species_list.value_counts()  # Frequency of each species
    return entropy(species_counts, base=base)

# calculate Shannon Index for each hexagon
shannon_indices = gdf_joined.groupby('hex_id')['species'].apply(compute_shannon_index).reset_index()
shannon_indices.rename(columns={'species': 'shannon_index'}, inplace=True)

# Merge the Shannon Index with hexagonal grid
hex_grid_shannon = hex_grid_gdf.merge(shannon_indices, on='hex_id', how='left')

# Handle missing values (hexagons with no species get Shannon Index = 0)
hex_grid_shannon['shannon_index'] = pd.to_numeric(hex_grid_shannon['shannon_index'], errors='coerce').fillna(0)

# Calculate areas for weighting and density metrics
hex_grid_shannon['area_total'] = hex_grid_shannon.geometry.area  # m²
hex_grid_shannon['area_within_swiss'] = hex_grid_shannon.intersection(swiss_border_gdf.unary_union).area  # m²
hex_grid_shannon['area_within_swiss_km2'] = hex_grid_shannon['area_within_swiss'] / 1_000_000  # km²
hex_grid_shannon['area_share'] = (
    hex_grid_shannon['area_within_swiss'] / hex_grid_shannon['area_total']).clip(0, 1)

# Calculate weighted Shannon Index (adjusted for partial hexagons on borders)
hex_grid_shannon['shannon_weighted'] = (
    hex_grid_shannon['shannon_index'] * hex_grid_shannon['area_share']
)

# Calculate Shannon density (index per km² inside Switzerland)
hex_grid_shannon['shannon_density_km2'] = (
    hex_grid_shannon['shannon_index'] / hex_grid_shannon['area_within_swiss_km2']
).replace([np.inf, -np.inf], 0).fillna(0)


'''not relevant for further analyis'''

'''Simpson Index'''

# Simpson Index calculation
def compute_simpson_index(species_list):
    """Compute Simpson Index (1 - D) and Inverse Simpson Index (1/D)."""
    if len(species_list) <= 1 or species_list.isna().all():
        return 0, 1  # Simpson (1 - D) = 0, Inverse = 1 for no diversity
    species_counts = species_list.value_counts()
    total = species_counts.sum()
    p = species_counts / total  # Proportions
    D = np.sum(p**2)  # Simpson's dominance index
    simpson_diversity = 1 - D  # Simpson Diversity Index
    inverse_simpson = 1 / D if D > 0 else 1  # Inverse Simpson Index
    return simpson_diversity, inverse_simpson

# Calculate Simpson Index per hexagon
simpson_results = gdf_joined.groupby('hex_id')['species'].apply(
    lambda x: compute_simpson_index(x)
).reset_index()
simpson_results[['simpson_index', 'inverse_simpson']] = pd.DataFrame(
    simpson_results['species'].tolist(), index=simpson_results.index
)
simpson_results.drop(columns='species', inplace=True)

# Merge Simpson Indices with hex grid
hex_grid_simpson = hex_grid_gdf.merge(simpson_results, on='hex_id', how='left')

# Handle missing values
for col in ['simpson_index', 'inverse_simpson']:
    hex_grid_simpson[col] = pd.to_numeric(hex_grid_simpson[col], errors='coerce').fillna(0)

# Calculate areas for weighting/normalization
hex_grid_simpson['area_total'] = hex_grid_simpson.geometry.area  # m²
hex_grid_simpson['area_within_swiss'] = hex_grid_simpson.intersection(swiss_border_gdf.unary_union).area  # m²
hex_grid_simpson['area_within_swiss_km2'] = hex_grid_simpson['area_within_swiss'] / 1_000_000  # km²
hex_grid_simpson['area_share'] = (
    hex_grid_simpson['area_within_swiss'] / hex_grid_simpson['area_total']
).clip(0, 1)

# Weighted Simpson Index
hex_grid_simpson['simpson_weighted'] = (
    hex_grid_simpson['simpson_index'] * hex_grid_simpson['area_share']
)

# Simpson Density (per km²)
hex_grid_simpson['simpson_density_km2'] = (
    hex_grid_simpson['simpson_index'] / hex_grid_simpson['area_within_swiss_km2']
).replace([np.inf, -np.inf], 0).fillna(0)



'''Comparison hex_id 166 & hex_id 167'''

# Filter for hex_id 166 and 167
hex_166_167 = gdf_joined[gdf_joined['hex_id'].isin([166, 167])]

# Merge richness and Shannon
hex_analysis = species_richness.merge(shannon_indices, on='hex_id', how='left')

# Species composition (counts per species)
species_counts = hex_166_167.groupby('hex_id')['species'].value_counts().unstack(fill_value=0)

# Merge with hex grid for geometry (optional)
hex_grid_analysis = hex_grid_gdf[hex_grid_gdf['hex_id'].isin([166, 167])].merge(
    hex_analysis, on='hex_id', how='left'
)

# Add area for context
hex_grid_analysis['area_within_swiss_km2'] = hex_grid_analysis.intersection(swiss_border_gdf.unary_union).area / 1_000_000

# Print summary
print("Hexagon Comparison (Richness and Shannon):")
print(hex_grid_analysis[['hex_id', 'species_richness', 'shannon_index', 'area_within_swiss_km2']])

print("\nSpecies Counts for Hex 166:")
print(species_counts.loc[166].sort_values(ascending=False).head(10))
print("\nSpecies Counts for Hex 167:")
print(species_counts.loc[167].sort_values(ascending=False).head(10))


# Visualization: Species abundance distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Hex 166
counts_166 = species_counts.loc[166].sort_values(ascending=False)
ax1.bar(range(len(counts_166)), counts_166)
ax1.set_title(f"Hex 166 (Richness: 101, Shannon: {shannon_indices[shannon_indices['hex_id'] == 166]['shannon_index'].values[0]:.2f})")
ax1.set_xlabel("Species Rank")
ax1.set_ylabel("Number of Occurrences")

# Hex 167
counts_167 = species_counts.loc[167].sort_values(ascending=False)
ax2.bar(range(len(counts_167)), counts_167)
ax2.set_title(f"Hex 167 (Richness: 101, Shannon: {shannon_indices[shannon_indices['hex_id'] == 167]['shannon_index'].values[0]:.2f})")
ax2.set_xlabel("Species Rank")

plt.tight_layout()
plt.savefig("/home/ubuntu/master/output/hex_166_167_comparison.png", dpi=300, bbox_inches='tight')
print("Plot save")
