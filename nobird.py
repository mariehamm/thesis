import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from scipy.stats import gaussian_kde

from scipy.stats import entropy

import vertebrates
import geometries

# Access the DataFrames from vertebrates & geometries module
mammals = vertebrates.mammals
amphibians = vertebrates.amphibians
reptiles = vertebrates.reptiles
hex_grid_gdf = geometries.hex_grid_gdf

## merge dataset mammal, amphibian, reptile (exclude birds)
combined_data = pd.concat([
    mammals[['class', 'species', 'decimalLatitude', 'decimalLongitude']],
    amphibians[['class', 'species', 'decimalLatitude', 'decimalLongitude']],
    reptiles[['class', 'species', 'decimalLatitude', 'decimalLongitude']]
])

combined_data.reset_index(drop=True, inplace=True)

# Convert the data to GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(combined_data['decimalLongitude'], combined_data['decimalLatitude'])]
gdf_species = gpd.GeoDataFrame(combined_data, geometry=geometry)

# Set CRS to WGS84 (EPSG:4326) initially
gdf_species.set_crs('EPSG:4326', allow_override=True, inplace=True)

# Reproject to LV95 (EPSG:2056)
gdf_species = gdf_species.to_crs(epsg=2056)
#print (gdf_species.head)

# Perform spatial join: find which hexagon each species point falls in
gdf_joined = gpd.sjoin(gdf_species, hex_grid_gdf, how='left', predicate='within')

## Species Points Count: this counts the total point observation per hexagon grid cell (does not consider "class")
# Count unique species points in each hexagon
species_points_count = gdf_joined.groupby('hex_id').size().reset_index(name='species_points_count')

# Merge the species points count with the hexagonal grid
hex_grid_with_count = hex_grid_gdf.merge(species_points_count, on='hex_id', how='left')

# Handle missing values (some hexagons may not have any species points, so we'll set count to 0)
hex_grid_with_count['species_points_count'] = hex_grid_with_count['species_points_count'].fillna(0)

# Save as GeoJSON
hex_grid_with_count.to_file("/home/ubuntu/master/output/species_points_count_3000_nb.geojson", driver='GeoJSON')


## ACTUAL Species Richness

# Count the number of unique species per hexagon
species_richness = gdf_joined.groupby('hex_id')['species'].nunique().reset_index()
species_richness.rename(columns={'species': 'species_richness'}, inplace=True)

# Create a column listing unique species per hexagon
species_names = gdf_joined.groupby('hex_id')['species'].apply(lambda x: ', '.join(sorted(set(x.dropna())))).reset_index()
species_names.rename(columns={'species': 'species_list'}, inplace=True)

# Create a column listing unique classes per hexagon
class_names = gdf_joined.groupby('hex_id')['class'].apply(lambda x: ', '.join(sorted(set(x.dropna())))).reset_index()
class_names.rename(columns={'class': 'class_list'}, inplace=True)

# Merge all data into the hexagon grid
hex_grid_species_richness = hex_grid_gdf.merge(species_richness, on='hex_id', how='left')
hex_grid_species_richness = hex_grid_species_richness.merge(species_names, on='hex_id', how='left')
hex_grid_species_richness = hex_grid_species_richness.merge(class_names, on='hex_id', how='left')

# Handle missing values (fill empty cells with 0 or empty strings)
hex_grid_species_richness['species_richness'] = hex_grid_species_richness['species_richness'].fillna(0)
hex_grid_species_richness['species_list'] = hex_grid_species_richness['species_list'].fillna('')
hex_grid_species_richness['class_list'] = hex_grid_species_richness['class_list'].fillna('')

# Save to GeoJSON
hex_grid_species_richness.to_file("/home/ubuntu/master/output/species_richness_3000_nb.geojson", driver='GeoJSON')


# Assuming combined_data contains your observation data with 'decimalLatitude', 'decimalLongitude', and 'species'
# Convert the observation data into a GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(combined_data['decimalLongitude'], combined_data['decimalLatitude'])]
gdf_diversity = gpd.GeoDataFrame(combined_data, geometry=geometry)

# Set the CRS 
gdf_diversity.set_crs(epsg=4326, allow_override=True, inplace=True)
gdf_diversity = gdf_diversity.to_crs(epsg=2056)

# join to assign species to hexagons
gdf_joined = gpd.sjoin(gdf_diversity, hex_grid_gdf, how='left', predicate='within')

# function to compute the Shannon Index
def compute_shannon_index(species_list):
    species_counts = species_list.value_counts()
    shannon_index = entropy(species_counts)  # Shannon Index
    return shannon_index

# Group by hex_id and calculate Shannon Index
shannon_indices = gdf_joined.groupby('hex_id')['species'].apply(compute_shannon_index).reset_index()
shannon_indices.rename(columns={'species': 'shannon_index'}, inplace=True)

# Merge the Shannon Index back to the hexagonal grid
hex_grid_with_shannon = hex_grid_gdf.merge(shannon_indices, on='hex_id', how='left')

# Handle missing values in Shannon Index (if any hexagon has no species, it will have a NaN)
hex_grid_with_shannon['shannon_index'] = hex_grid_with_shannon['shannon_index'].fillna(0)

hex_grid_with_shannon.to_file("/home/ubuntu/master/output/shannon_index_hex3000_nb.geojson", driver='GeoJSON')

