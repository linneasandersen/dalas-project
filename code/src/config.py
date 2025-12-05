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


COUNTRIES_ALT = {
    "Korea, Rep.": "South Korea",
    "Egypt, Arab Rep.": "Egypt",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Turkiye": "Turkey",
    "Russian Federation": "Russia",
    "Slovak Republic": "Slovakia"
}