from pathlib import Path

possible_paths = [
    Path("/Users/linneaandersen/Library/CloudStorage/GoogleDrive-linneasandersen@gmail.com/Mit drev/DALAS/data"), # Modern
    Path("/Users/linneaandersen/Google Drive/Mit drev/DALAS/data"), # Legacy
    Path.home() / "Google Drive" / "Mit drev" / "DALAS" / "data" # Dynamic Home
]

# 2. Select the first one that actually exists
GOOGLE_DRIVE = next((p for p in possible_paths if p.exists()), None)
GOOGLE_DRIVE_EN = Path("/Users/linneaandersen/Google Drive/My Drive/DALAS/data")

RAW_DIR = GOOGLE_DRIVE / "raw"
PROCESSED_DIR = GOOGLE_DRIVE / "processed"

LOCAL_DIR = Path(__file__).parent.parent.parent.parent / "dalas-data"

RAW_DIR_EN = GOOGLE_DRIVE_EN / "raw"
PROCESSED_DIR_EN = GOOGLE_DRIVE_EN / "processed"
MERGED_DIR = PROCESSED_DIR / "merged"
EDA_DIR = PROCESSED_DIR / "eda"
LAGGED_DIR = PROCESSED_DIR / "train"

YEARS = range(2010, 2024)
HS2_CODES = [10, 12, 15, 26, 27, 29, 31, 39, 71, 72, 73, 75, 76, 81, 84, 85, 87, 88, 90]
#OEC_CODES = [210, 212, 315, 526, 527, 629, 631, 739, 1471, 1572, 1573, 1575, 1576, 1581, 1684, 1685, 1787, 1788, 1890]
OEC_CODES = [210]
COUNTRIES = [
    "Ukraine",
    "Russia",
    "Germany",
    "Poland",
    "Lithuania",
    "Latvia",
    "Estonia",
    "Hungary",
    "Slovakia",
    "Italy",
    "France",
    "United Kingdom",
    "China",
    "India",
    "Turkey",
    "Japan",
    "South Korea",
    "Egypt",
    "Bangladesh",
    "Indonesia",
    "Pakistan",
    "Lebanon",
    "Tunisia",
    "Morocco",
    "Brazil",
    "United States",
    "Canada",
    "Armenia",
    "Kazakhstan",
    "Kyrgyzstan",
    "United Arab Emirates",
    "Georgia",
    "Romania",
    "Bulgaria",
    "Singapore"
]

#REGIONS = ["Europe", "Asia", "Africa", "Middle" "South America", "North America"]
WB_REGIONS = ["East Asia and Pacific", "Europe and Central Asia", "Latin America and Caribbean", "Middle East, North Africa, Afghanistan and Pakistan", "North America", "South Asia", "Sub-Saharan Africa"]
CONTINENTAL_REGIONS = ["Asia", "Africa", "North America", "South America", "Antarctica", "Europe", "Australia"]
# label countries with a continental region
COUNTRIES_REGIONS = {
    # Europe and Central Asia → Europe or Asia
    "Ukraine": "Europe",
    "Russia": "Europe",  # mostly Europe part for simplicity
    "Germany": "Europe",
    "Poland": "Europe",
    "Lithuania": "Europe",
    "Latvia": "Europe",
    "Estonia": "Europe",
    "Hungary": "Europe",
    "Slovakia": "Europe",
    "Italy": "Europe",
    "France": "Europe",
    "United Kingdom": "Europe",
    "Turkey": "Asia",  # Anatolian part
    "Armenia": "Asia",
    "Kazakhstan": "Asia",
    "Kyrgyzstan": "Asia",
    "Georgia": "Asia",
    "Romania": "Europe",
    "Bulgaria": "Europe",

    # East Asia and Pacific → Asia or Australia
    "China": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "Indonesia": "Asia",
    "Singapore": "Asia",

    # South Asia → Asia
    "India": "Asia",
    "Bangladesh": "Asia",
    "Pakistan": "Asia",

    # Middle East, North Africa, Afghanistan and Pakistan → Asia or Africa
    "Egypt": "Africa",
    "Lebanon": "Asia",
    "Tunisia": "Africa",
    "Morocco": "Africa",
    "United Arab Emirates": "Asia",

    # Latin America and Caribbean → South America
    "Brazil": "South America",

    # North America
    "United States": "North America",
    "Canada": "North America",
}



COUNTRIES_REGIONS_WB = {
    "Ukraine": "Europe and Central Asia",
    "Russia": "Europe and Central Asia",
    "Germany": "Europe and Central Asia",
    "Poland": "Europe and Central Asia",
    "Lithuania": "Europe and Central Asia",
    "Latvia": "Europe and Central Asia",
    "Estonia": "Europe and Central Asia",
    "Hungary": "Europe and Central Asia",
    "Slovakia": "Europe and Central Asia",
    "Italy": "Europe and Central Asia",
    "France": "Europe and Central Asia",
    "United Kingdom": "Europe and Central Asia",
    "China": "East Asia and Pacific",
    "India": "South Asia",
    "Turkey": "Europe and Central Asia",
    "Japan": "East Asia and Pacific",
    "South Korea": "East Asia and Pacific",
    "Egypt": "Middle East, North Africa, Afghanistan and Pakistan",
    "Bangladesh": "South Asia",
    "Indonesia": "East Asia and Pacific",
    "Pakistan": "Middle East, North Africa, Afghanistan and Pakistan",
    "Lebanon": "Middle East, North Africa, Afghanistan and Pakistan",
    "Tunisia": "Middle East, North Africa, Afghanistan and Pakistan",
    "Morocco": "Middle East, North Africa, Afghanistan and Pakistan",
    "Brazil": "Latin America and Caribbean",
    "United States": "North America",
    "Canada": "North America",
    "Armenia": "Europe and Central Asia",
    "Kazakhstan": "Europe and Central Asia",
    "Kyrgyzstan": "Europe and Central Asia",
    "United Arab Emirates": "Middle East, North Africa, Afghanistan and Pakistan",
    "Georgia": "Europe and Central Asia",
    "Romania": "Europe and Central Asia",
    "Bulgaria":  "Europe and Central Asia",
    "Singapore": "East Asia and Pacific"
}

COUNTRIES_ALT = {
    "Korea, Rep.": "South Korea",
    "Korea (the Republic of)": "South Korea",
    "Republic of Korea": "South Korea",
    "United Arab Emirates (the)": "United Arab Emirates",
    "Russian Federation (the)": "Russia",
    "United States of America (the)": "United States",
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland (the)": "United Kingdom",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Egypt, Arab Rep.": "Egypt",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Russian Federation": "Russia",
    "Slovak Republic": "Slovakia"
}


# https://raw.githubusercontent.com/google/dspl/master/samples/google/canonical/countries.csv
# https://www.ewf.uni-bayreuth.de/en/research/RTA-data/index.html

PRODUCT_NAMES = {
    10: "Cereals",
    12: "Oils seeds, oleaginous fruits, grains, straw and fodder",
    15: "Animal or vegetable fats, oils, and waxes",
    26: "Ores, slag and ash",
    27: "Mineral fuels, mineral oils and products of their distillation",
    29: "Organic chemicals",
    31: "Fertilizers",
    39: "Plastics and articles thereof",
    71: "Precious stones, metals, and pearls",
    72: "Iron and steel",
    73: "Iron/Steel articles",
    75: "Nickel articles",
    76: "Aluminum articles",
    81: "Cermet articles",
    84: "Machinery, mechanical appliances, and parts",
    85: "Electrical machinery and electronics",
    87: "Cars, tractors, trucks and parts thereof",
    88: "Aircraft and Spacecraft",
    90: "Optical, photo, and film equipment; medical instruments"
}

PRODUCT_NAMES_SHORT = {
    10: "Cereals",
    12: "Oils seeds",
    15: "Animal or vegetable fats",
    26: "Ores, slag and ash",
    27: "Mineral fuels and oils",
    29: "Organic chemicals",
    31: "Fertilizers",
    39: "Plastics",
    71: "Precious stones, metals, and pearls",
    72: "Iron and steel",
    73: "Iron/steel articles",
    75: "Nickel articles",
    76: "Aluminum articles",
    81: "Cermet articles",
    84: "Machinery, mechanical",
    85: "Electrical machinery, and electronics",
    87: "Cars, tractors, and trucks",
    88: "Aircraft and spacecraft",
    90: "Medical instruments"
}


# 1. Agricultural
AGRICULTURAL_HS2_CODES = [10, 12, 15]

# 2. Minerals, Energy & Metals
MINERALS_HS2_CODES = [26, 27]
METALS_HS2_CODES = [71, 72, 73, 75, 76, 81]

# 3. Chemicals & Intermediate Inputs
CHEMICALS_HS2_CODES = [29, 31]  # 29 = organic chemicals, 31 = fertilizers

# 4. Manufactured Goods (Machinery, Electronics, Vehicles, Aircraft, etc.)
MANUFACTURED_HS2_CODES = [39, 84, 85, 87, 88, 90]

PRODUCT_CATEGORIES = {
    **{code: "Agricultural" for code in AGRICULTURAL_HS2_CODES},
    **{code: "Minerals" for code in MINERALS_HS2_CODES},
    **{code: "Metals" for code in METALS_HS2_CODES},
    **{code: "Chemicals" for code in CHEMICALS_HS2_CODES},
    **{code: "Manufactured" for code in MANUFACTURED_HS2_CODES},
}


PRODUCT_PALETTE = {
    10: {"tertiary": "#FFF176", "secondary": "#FFECB3", "primary": "#F9A825"},  # Cereals
    12: {"tertiary": "#81C784", "secondary": "#C8E6C9", "primary": "#2E7D32"},  # Seeds
    15: {"tertiary": "#FFB74D", "secondary": "#FFE0B2", "primary": "#E65100"},  # Fats
    26: {"tertiary": "#B0BEC5", "secondary": "#ECEFF1", "primary": "#6AA9C8"},  # Ores
    27: {"tertiary": "#9E9E9E", "secondary": "#E0E0E0", "primary": "#212121"},  # Fuels
    29: {"tertiary": "#BA68C8", "secondary": "#E1BEE7", "primary": "#6A1B9A"},  # Organics
    31: {"tertiary": "#4DB6AC", "secondary": "#B2DFDB", "primary": "#00695C"},  # Fertilizers
    39: {"tertiary": "#FF8A65", "secondary": "#FFCCBC", "primary": "#D84315"},  # Plastics
    71: {"tertiary": "#FFE066", "secondary": "#FFF9C4", "primary": "#FFB300"},  # Precious
    72: {"tertiary": "#78909C", "secondary": "#CFD8DC", "primary": "#263238"},  # Iron/Steel
    73: {"tertiary": "#90A4AE", "secondary": "#ECEFF1", "primary": "#37474F"},  # Iron Art.
    75: {"tertiary": "#BCAAA4", "secondary": "#D7CCC8", "primary": "#5D4037"},  # Nickel
    76: {"tertiary": "#CFD8DC", "secondary": "#F5F5F5", "primary": "#607D8B"},  # Aluminum
    81: {"tertiary": "#78909C", "secondary": "#ECEFF1", "primary": "#263238"},  # Cermet
    84: {"tertiary": "#64B5F6", "secondary": "#BBDEFB", "primary": "#0D47A1"},  # Machinery
    85: {"tertiary": "#4DD0E1", "secondary": "#B2EBF2", "primary": "#006064"},  # Electronics
    87: {"tertiary": "#EF5350", "secondary": "#FFCDD2", "primary": "#B71C1C"},  # Vehicles
    88: {"tertiary": "#42A5F5", "secondary": "#BBDEFB", "primary": "#0D47A1"},  # Aircraft
    90: {"tertiary": "#26A69A", "secondary": "#B2DFDB", "primary": "#004D40"}   # Optical
}
