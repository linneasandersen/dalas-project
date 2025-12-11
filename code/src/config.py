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
    "Egypt, Arab Rep.": "Egypt",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Turkiye": "Turkey",
    "Russian Federation": "Russia",
    "Slovak Republic": "Slovakia"
}