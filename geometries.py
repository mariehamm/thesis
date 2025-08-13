''' This Python script generates a hexagonal grid overlaying the geographical boundary of Switzerland 
using the SwissBOUNDARIES3D dataset in the LV95 (CH1903+/CRS 2056) coordinate reference system. 
The grid consists of regular hexagons with a specified side length (7500 meters), 
and only hexagons intersecting Switzerland's boundary are retained. 
The output is a GeoDataFrame.
The script is based on a hexagonal grid algorithm inspired by GIS Stack Exchange 
and references: https://gis.stackexchange.com/questions/341218/creating-a-hexagonal-grid-of-regular-hexagons-of-definite-area-anywhere-on-the-g

'''

# import libraries
import matplotlib.pyplot as plt
import geopandas as gpd
import fiona
import os
import numpy as np
from shapely.geometry import Polygon
import math

# Function to create a single hexagon centered at (x, y) with side length l
# Based on: https://gis.stackexchange.com/questions/341218/creating-a-hexagonal-grid-of-regular-hexagons-of-definite-area-anywhere-on-the-g

def create_hexagon(l, x, y):
    """
    Create a hexagon centered on (x, y)
    :param l: length of the hexagon's edge
    :param x: x-coordinate of the hexagon's center
    :param y: y-coordinate of the hexagon's center
    :return: The polygon containing the hexagon's coordinates
    """
    c = [[x + math.cos(math.radians(angle)) * l, y + math.sin(math.radians(angle)) * l] for angle in range(0, 360, 60)]
    return Polygon(c)

# Function to generate a grid of hexagon centers within a bounding box

def create_hexgrid(bbox, side, offset_x=0, offset_y=0):
    """
    Returns an array of Points describing hexagon centers that are inside the given bounding_box.
    :param bbox: The containing bounding box. The bbox coordinates should be in the same CRS as the hexagons.
    :param side: The size of the hexagon's side
    :return: The hexagon grid
    """
    grid = []
    v_step = math.sqrt(3) * side
    h_step = 1.5 * side

    x_min = min(bbox[0], bbox[2])
    x_max = max(bbox[0], bbox[2])
    y_min = min(bbox[1], bbox[3])
    y_max = max(bbox[1], bbox[3])

    h_skip = math.ceil((x_min + offset_x) / h_step) - 1
    h_start = h_skip * h_step

    v_skip = math.ceil((y_min + offset_y) / v_step) - 1
    v_start = v_skip * v_step

    h_end = x_max + h_step
    v_end = y_max + v_step
    
    if v_start - (v_step / 2.0) < y_min:
        v_start_array = [v_start + (v_step / 2.0), v_start]
    else:
        v_start_array = [v_start - (v_step / 2.0), v_start]

    v_start_idx = int(abs(h_skip) % 2)

    c_x = h_start
    c_y = v_start_array[v_start_idx]
    v_start_idx = (v_start_idx + 1) % 2
    while c_x < h_end:
        while c_y < v_end:
            grid.append((c_x, c_y))
            c_y += v_step
        c_x += h_step
        c_y = v_start_array[v_start_idx]
        v_start_idx = (v_start_idx + 1) % 2

    return grid

# Load the Swiss boundary shapefile and transform to CRS 2056
# Shapefile source: SwissBOUNDARIES3D from swisstopo
swiss_boundary = gpd.read_file("swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp")
swiss_boundary = swiss_boundary.to_crs(2056)

# Define columns to keep: name, population, and geometry
keep_columns = [
    'NAME', 'EINWOHNERZ', 'geometry'
]
swiss_boundary = swiss_boundary[keep_columns]
swiss_boundary = swiss_boundary[swiss_boundary['NAME'] == 'Schweiz']

# Get bounding box coordinates of Switzerland
minx, miny, maxx, maxy = swiss_boundary.total_bounds

# Define hexagon side length 
# https://datacore-gn.unepgrid.ch/geonetwork/srv/api/records/f0f80450-83a1-4fc8-95e9-34e23f688fed
# https://www.kontur.io/blog/why-we-use-h3/
hex_size = 7500  # Length of hexagon side (works best for BD Data)

# Generate hexagonal grid using the bounding box
hex_centers = create_hexgrid((minx, miny, maxx, maxy), hex_size)

# Create hexagons from the grid of centers and filter by intersection 
hexagons = []
for (x, y) in hex_centers:
    hexagon = create_hexagon(hex_size, x, y)
    
    # Keep only hexagons that intersect with the boundary
    if swiss_boundary.geometry.unary_union.intersects(hexagon):
        hexagons.append(hexagon)

# Convert to GeoDataFrame
hex_grid_gdf = gpd.GeoDataFrame(geometry=hexagons, crs=swiss_boundary.crs)
hex_grid_gdf["hex_id"] = range(1, len(hex_grid_gdf) + 1)

# Save or display the result
#hex_grid_gdf.to_file("/home/ubuntu/master/output/hex_grid_10000.geojson", driver="GeoJSON")
#print(hex_grid_gdf)

