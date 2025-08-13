import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape
from rasterio.features import geometry_mask
import numpy as np
from rasterio.features import sieve 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import jenkspy 

from rasterstats import zonal_stats
import plotly.graph_objects as go
import subprocess


import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from collections import Counter


import geometries

swiss_boundary = geometries.swiss_boundary


'''
# Print basic stats
print("Data type:", raster_data.dtype)
print("Min value:", np.min(raster_data))
print("Max value:", np.max(raster_data))
print("Contains NaN:", np.any(np.isnan(raster_data)))
print("Contains Inf:", np.any(np.isinf(raster_data)))
'''


# Open the original raster
with rasterio.open("/home/ubuntu/master/data/AHP_0318.tif") as src:
    raster_data = src.read(1, masked=True)
    meta = src.meta.copy()

# Define NoData value from the original raster
original_nodata = -3.4028235e+38
# Mask NoData explicitly
raster_data = np.ma.masked_where(raster_data == original_nodata, raster_data)

valid_data = raster_data [~raster_data.mask].flatten()
print("total valid data points:", len(valid_data))

# Sample a subset for Jenks 
sample_size = min(10000, len(valid_data))  
if sample_size < len(valid_data):
    sample_data = np.random.choice(valid_data, size=sample_size, replace=False)
else:
    sample_data = valid_data
print("Sample size for Jenks:", len(sample_data))

jenks_breaks = jenkspy.jenks_breaks(sample_data, n_classes= 5)
print("Jenks breaks:", jenks_breaks)

'''
Sample size for Jenks: 10000
Jenks breaks: [0.21727246, 0.49630132, 0.61153054, 0.7056344, 0.786755, 0.9191844]

Sample size for Jenks: 100000
Jenks breaks: [0.13068219, 0.49542245, 0.61238825, 0.7066963, 0.7872156, 0.9187055]

'''

'''
#Plot histogram with Jenks breaks
plt.figure(figsize=(10, 6))
plt.hist(sample_data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
for break_val in jenks_breaks[1:-1]:  # Skip min and max
    plt.axvline(x=break_val, color='red', linestyle='--', label=f'Jenks Break: {break_val:.3f}')
plt.title('Histogram of Raster Data with Jenks Natural Breaks')
plt.xlabel('Biodiversity Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('jenks_histogram.png')
print("Histogram saved as jenks_histogram.png")
plt.close()
'''

'''

# Reclassify the raster
classified_data = np.digitize(raster_data.filled(original_nodata), jenks_breaks[:-1], right=True)
classified_data = np.where(raster_data.mask, -1, classified_data)  # Use -1 as new NoData

# Update metadata
meta.update(dtype=rasterio.int8, nodata=-1)  # Set NoData to -1

# Save the classified raster

with rasterio.open("/home/ubuntu/master/data/AHP_0318_classified.tif", "w", **meta) as dst:
    dst.write(classified_data.astype(rasterio.int8), 1)

#print("Classified raster saved as AHP_0318_classified.tif")

## --- Polygonize classified Raster ---

#import rasterio
from rasterio.features import sieve
import numpy as np
import subprocess

# Paths
classified_raster_path = "/home/ubuntu/master/data/AHP_0318_classified.tif"
sieved_raster_path = "/home/ubuntu/master/data/AHP_0318_sieved.tif"
raw_polygons_path = "/home/ubuntu/master/data/AHP_0318_polygons_raw.gpkg"
final_polygons_path = "/home/ubuntu/master/data/AHP_0318_classified_polygons.gpkg"

# Step 1: Sieve the raster
with rasterio.open(classified_raster_path) as src:
    classified_data = src.read(1)
    meta = src.meta.copy()
    nodata = src.nodata

    # Convert to a supported data type (uint8)
    classified_data = classified_data.astype(np.uint8)

    # Apply sieve: remove regions smaller than 100 pixels
    min_size = 100  # Adjust as needed
    sieved_data = sieve(classified_data, size=min_size, connectivity=8)

    # Save the sieved raster
    with rasterio.open(sieved_raster_path, 'w', **meta) as dst:
        dst.write(sieved_data, 1)

print(f"Sieved raster saved as {sieved_raster_path}")

# Step 2: Polygonize using gdal_polygonize.py
subprocess.run([
    "gdal_polygonize.py",
    sieved_raster_path,
    "-b", "1",
    "-f", "GPKG",
    raw_polygons_path,
    "AHP_0318",
    "class"
], check=True)
print(f"Raw polygons saved as {raw_polygons_path}")

# Step 3: Dissolve polygons by class using ogr2ogr
subprocess.run([
    "ogr2ogr",
    final_polygons_path,
    raw_polygons_path,
    "-sql", "SELECT class, ST_Union(geom) as geom FROM AHP_0318 GROUP BY class",
    "-dialect", "SQLITE"
], check=True)
print(f"Classified polygons saved as {final_polygons_path}")

'''




## --- Sankey Diagramm Preparation---

# --- Load and Preprocess Data: Swiss Boundary & ARE Priorities ---
swiss = gpd.read_file("/home/ubuntu/master/data/Swiss Boundary LV95/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp")
pv_polygons = gpd.read_file("/home/ubuntu/master/data/prioritiesARE_diss.gpkg")
swiss = swiss.to_crs(epsg=2056)
pv_polygons = pv_polygons.to_crs(epsg=2056)

# simplify geometry to reduce load
pv_polygons["geometry"] = pv_polygons.simplify(tolerance=5, preserve_topology=True)



# Spatial Join after merging
pv_swiss = gpd.sjoin(pv_polygons, swiss, how="inner", predicate="intersects")

pv_swiss.reset_index(drop=True, inplace=True)



# --- Extract Classified Biodiversity Values ---
stats = zonal_stats(
    pv_swiss,
    "/home/ubuntu/master/data/AHP_0318_classified.tif",
    stats="majority",
    nodata=-1  # Matches the NoData from the classified raster
)
pv_swiss['biodiv_class'] = [stat['majority'] if stat else np.nan for stat in stats]

# Map to readable categories
class_map = {1: 'Very low Biodiversity Value', 
             2: 'Low Biodiversity Value', 
             3: 'Medium Biodiversity Value', 
             4: 'High Biodiversity Value', 
             5: "Very high Biodiversity Value"}

pv_swiss["biodiv_category"] = pv_swiss["biodiv_class"].map(class_map)

pv_swiss = pv_swiss.dropna(subset=['biodiv_category'])

pv_swiss.reset_index(drop=True, inplace=True)

suitability_map = {
    4: "Priority 4 (other areas worth examining)", 
    3: "Priority 3", 
    2: "Priority 2", 
    1: "Priority 1"
}

pv_swiss['suitability_label'] = pv_swiss['priority'].map(suitability_map)


# Debug: Check mapping results
print("Unique biodiv_category values:", pv_swiss['biodiv_category'].unique())
print("Unique suitability_label values:", pv_swiss['suitability_label'].unique())
print("Any NaN in suitability_label?", pv_swiss['suitability_label'].isna().any())
print("Any NaN in biodiv_category?", pv_swiss['biodiv_category'].isna().any())

# --- Aggregate for Sankey ---
agg_data = pv_swiss.groupby(['suitability_label', 'biodiv_category']).size().reset_index(name='count')

#print("Aggregated data:\n", agg_data)'''



#Calculate each Percentages

total_count = agg_data['count'].sum()
print(f"Total sites: {total_count}")

'''
# 1. Check: poor to low suitability
poor_low_suit = agg_data[agg_data['suitability_label'].isin(['Priority 4 (other areas worth examining)', 'Priority 3'])]['count'].sum()
poor_low_suit_pct = (poor_low_suit / total_count) * 100
print(f"Poor to low suitability sites: {poor_low_suit} ({poor_low_suit_pct:.1f}% of total)")

# 2. Check: how much of poor/low suitability sites are high/very high biodiversity
poor_low_high_biodiv = agg_data[
    (agg_data['suitability_label'].isin(['Priority 4 (other areas worth examining)', 'Priority 3'])) &
    (agg_data['biodiv_category'].isin(['High Biodiversity Value', 'Very High Biodiversity Value']))
]['count'].sum()
poor_low_high_biodiv_pct = (poor_low_high_biodiv / poor_low_suit) * 100 if poor_low_suit > 0 else 0
print(f"Poor/low suitability sites in high/very high biodiversity: {poor_low_high_biodiv} ({poor_low_high_biodiv_pct:.1f}% of poor/low suitability)")

# 3. Check: how much of highly suitable sites align with medium biodiversity
high_suit = agg_data[agg_data['suitability_label'] == 'Priority 2']['count'].sum()
high_suit_medium_biodiv = agg_data[
    (agg_data['suitability_label'] == 'Priority 2') &
    (agg_data['biodiv_category'] == 'Medium Biodiversity Value')
]['count'].sum()
high_suit_medium_biodiv_pct = (high_suit_medium_biodiv / high_suit) * 100 if high_suit > 0 else 0
print(f"Highly suitable sites in medium biodiversity: {high_suit_medium_biodiv} ({high_suit_medium_biodiv_pct:.1f}% of highly suitable)")
'''

suitability_colors = {
    'Priority 1': 'rgba(161, 0, 161, 0.4)',    # Magenta/Purple
    'Priority 2': 'rgba(255, 193, 7, 0.4)',     # Yellow/Amber
    'Priority 3': 'rgba(0, 161, 161, 0.4)', # Cyan/Turquoise
    'Priority 4 (other areas worth examining)': 'rgba(74, 74, 74, 0.4)'      # Dark Gray
}



# --- Sankey Diagram ---
source, target, value, link_colors, link_labels = [], [], [], [], []

suitability_nodes = [
    "Priority 1",
    "Priority 2",
    "Priority 3",
    "Priority 4 (other areas worth examining)"
]

biodiv_nodes = [
    "Very high Biodiversity Value",
    "High Biodiversity Value",
    "Medium Biodiversity Value",
    "Low Biodiversity Value",
    "Very low Biodiversity Value"
]

all_nodes = suitability_nodes + biodiv_nodes
node_dict = {node: i for i, node in enumerate(all_nodes)}


for _, row in agg_data.iterrows():
    suitability_label = row['suitability_label']
    biodiv_label = row['biodiv_category']
    count = row['count']
    source.append(node_dict[suitability_label])
    target.append(node_dict[biodiv_label])
    value.append(count)
    link_colors.append(suitability_colors[suitability_label])
    # Calculate percentage for link label
    percentage = (count / total_count) * 100
    link_labels.append(f"{count} ({percentage:.1f}%)")

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color="gray"
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors,
        customdata=link_labels,
        hovertemplate='%{customdata}<extra></extra>'
    )
)])
fig.update_layout(
    title_text="PV Priority Classes per Biodiversity Classes in Switzerland",
    font_size=10
)
fig.show()


















