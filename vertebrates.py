'''This Python script loads and preprocesses vertebrate occurrence data (mammals, amphibians, reptiles, birds)
from CSV files. It cleans the datasets by removing unnecessary columns, adds
German translations for taxonomic orders, categorizes species by size group. 
The processed DataFrames are intended for use in geospatial analysis
(mapping species richness onto a hexagonal grid).'''

# import libraries
import pandas as pd
import os

'''Mammals data'''

mammals = pd.read_csv('/home/ubuntu/master/data/mammal data.csv', sep='\t', low_memory=False)

# Drop unnecessary columns
mammals.drop(columns=[
    'datasetKey', 'occurrenceID', 'infraspecificEpithet', 'taxonRank', 
    'verbatimScientificName', 'verbatimScientificNameAuthorship', 'locality', 
    'individualCount', 'publishingOrgKey', 'coordinatePrecision', 'depth', 
    'elevationAccuracy', 'depthAccuracy', 'eventDate', 'day', 'month', 
    'institutionCode', 'collectionCode', 'recordNumber', 'identifiedBy', 
    'dateIdentified', 'license', 'recordedBy', 'typeStatus', 'establishmentMeans', 
    'lastInterpreted', 'mediaType', 'issue', 'kingdom', 'phylum', 
    'taxonKey', 'catalogNumber', 'rightsHolder', 'speciesKey', 'basisOfRecord', 
    'countryCode', 'occurrenceStatus'
], inplace=True, errors = 'ignore')

# Add German order name
translations_m = {
    "Soricomorpha": "Spitzmausartige",
    "Rodentia": "Nagetiere",
    "Carnivora": "Raubtiere",
    "Artiodactyla": "Paarhufer",
    "Lagomorpha": "Hasenartige",
    "Erinaceomorpha": "Igelartige"
}

mammals['german_order'] = mammals['order'].map(translations_m)

# Reorder columns to place 'german_order' after 'order'
cols = mammals.columns.tolist()
cols.insert(cols.index("order") + 1, cols.pop(cols.index("german_order")))
mammals = mammals[cols]

# Categorize mammals by size group
size_mapping = {
    "Soricomorpha": "Small",
    "Rodentia": "Small",
    "Erinaceomorpha": "Medium",
    "Lagomorpha": "Medium"
}
mammals['Size_Group'] = mammals['order'].map(size_mapping).fillna("Large")
#print (mammals.head(4))


'''Amphibian data'''

amphibians = pd.read_csv('/home/ubuntu/master/data/amphibian data.csv', sep='\t', low_memory=False)

# Drop unnecessary columns
columns_to_drop = [
    'datasetKey', 'occurrenceID', 'infraspecificEpithet', 'taxonRank', 'verbatimScientificName',
    'verbatimScientificNameAuthorship', 'locality', 'individualCount', 'publishingOrgKey',
    'coordinatePrecision', 'depth', 'elevationAccuracy', 'depthAccuracy', 'eventDate', 'day', 'month',
    'institutionCode', 'collectionCode', 'recordNumber', 'identifiedBy', 'dateIdentified', 'license',
    'recordedBy', 'typeStatus', 'establishmentMeans', 'lastInterpreted', 'mediaType', 'issue', 'kingdom',
    'phylum', 'taxonKey', 'catalogNumber', 'rightsHolder', 'speciesKey', 'basisOfRecord',
    'countryCode', 'occurrenceStatus'
]
amphibians.drop(columns=columns_to_drop, inplace=True)

# Translations for amphibians
translations_a = {
    'Caudata': 'Schwanzlurche',
    'Anura': 'Froschlurche'
}

# Add a new column for the German translation
amphibians['german_order'] = amphibians['order'].map(translations_a)

# Relocate german_order column (optional, as Python does not need explicit relocation)
cols = list(amphibians.columns)
cols.insert(cols.index('order') + 1, cols.pop(cols.index('german_order')))
amphibians = amphibians[cols]

# Categorize both amphibian orders as "Small"
amphibians['Size_Group'] = "Small"


'''Reptiles data'''

reptiles = pd.read_csv('/home/ubuntu/master/data/reptile data.csv', sep='\t', low_memory=False)

# Drop unnecessary columns
drop_columns = ["datasetKey", "occurrenceID", "infraspecificEpithet", "taxonRank", "verbatimScientificName", 
                "verbatimScientificNameAuthorship", "locality", "individualCount", "publishingOrgKey", 
                "coordinatePrecision", "depth", "elevationAccuracy", "depthAccuracy", "eventDate", "day", 
                "month", "institutionCode", "collectionCode", "recordNumber", "identifiedBy", "dateIdentified", 
                "license", "recordedBy", "typeStatus", "establishmentMeans", "lastInterpreted", "mediaType", 
                "issue", "kingdom", "phylum", "taxonKey", "catalogNumber", "rightsHolder", "speciesKey", 
                "basisOfRecord", "countryCode", "occurrenceStatus"]




# Define German translations
translations_r = {
    "Squamata": "Schuppenkriechtiere",  
    "Testudines": "Schildkröten"
}

# Set the 'class' column to 'Reptilia'
reptiles['class'] = 'Reptilia'


# Add a new column for the German translation
reptiles['german_order'] = reptiles['order'].map(translations_r)

# Relocate the german_order column to appear after order
column_order = reptiles.columns.tolist()
column_order.insert(column_order.index("order") + 1, column_order.pop(column_order.index("german_order")))
reptiles = reptiles[column_order]


# Categorize all reptiles (including NaN in order) as "Small"
reptiles["Size_Group"] = "Small"


'''Breeding bird data'''
## Load breeding bird data

birds = pd.read_csv('/home/ubuntu/master/data/breedingbird data.csv', sep='\t', low_memory=False)

# List of columns you want to keep
keep_columns = [
    'class', 'order', 'family', 'genus', 'species', 'scientificName', 'locality', 
    'stateProvince', 'individualCount', 'decimalLongitude', 'decimalLatitude', 
    'elevation', 'coordinatePrecision', 'year'
]

# Select only the columns you want to keep
birds = birds[keep_columns]



'''not relevant for further analysis'''

'''Red list data'''

# Load the CSV file
redlist_mammal = pd.read_csv('/home/ubuntu/master/data/BAFU Red List Mammals.csv', sep=',', low_memory=False)

# Drop unnecessary columns
drop_columns = ["GROUP", "(F) Name", "(I) Name", "Remarques", "Annotazioni"]
redlist_mammal.drop(columns=drop_columns, inplace=True)


# Filter out species that are "locally not applicable (NA)" and "not endangered (LC)"
redlist_mammal = redlist_mammal[(redlist_mammal["CAT"] != "NA") & (redlist_mammal["CAT"] != "LC")]

# Define translations for red list categories
translations_rl = {
    "RE": "in CH ausgestorben",
    "CR": "vom Aussterben bedroht",
    "EN": "Stark gefährdet",
    "NT": "potenziell gefährdet",
    "DD": "ungenügende Datengrundlage",
    "NE": "nicht beurteilt",
    "VU": "verletzlich"
}

# Add a new column for the definition of Category
redlist_mammal["definition"] = redlist_mammal["CAT"].map(translations_rl)

# Relocate the definition column to appear after CAT
column_order = redlist_mammal.columns.tolist()
column_order.insert(column_order.index("CAT") + 1, column_order.pop(column_order.index("definition")))
redlist_mammal = redlist_mammal[column_order]

#print(redlist_mammal.head)



