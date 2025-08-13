# Biodiversity and Photovoltaic (PV) Priority Analysis in Switzerland

This repository contains Python scripts for analyzing biodiversity and photovoltaic (PV) priority areas in Switzerland using a hexagonal grid, the Analytic Hierarchy Process (AHP), and spatial data processing. The scripts calculate biodiversity indices, prioritize ecological criteria, classify biodiversity values, and visualize relationships between biodiversity and PV suitability using a Sankey diagram. The analysis is conducted in the Swiss LV95 coordinate system (EPSG:2056).

# Project Overview

The project consists of six main scripts:

ahp.py: Implements AHP to prioritize biodiversity and ecosystem function criteria, performs sensitivity analysis, and generates bar plots for global weights and sensitivity results.

diversity.py: Calculates Shannon and Simpson biodiversity indices for species occurrences on a hexagonal grid over Switzerland, including area-weighted and density metrics. Outputs GeoJSON files and a comparison plot for two hexagons.

geometries.py: Calculates hexagonal grid over Switzerland 

newsankey.py: Classifies a biodiversity raster using Jenks natural breaks, polygonizes it, and analyzes its overlap with PV priority areas, visualized via an interactive Sankey diagram.

richness.py: Calculates Species richness for species occurences on a hexagonal grid over Switzerland.

vertebrates.py: Prepares mammal, amphibian, reptile and bird data for further analyis. 




biodiversity_pv_analysis.py: 
