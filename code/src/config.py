from pathlib import Path

GOOGLE_DRIVE = Path("/Users/linneaandersen/Google Drive/Mit drev/DALAS/data")
GOOGLE_DRIVE_EN = Path("/Users/linneaandersen/Google Drive/My Drive/DALAS/data")

RAW_DIR = GOOGLE_DRIVE / "raw"
PROCESSED_DIR = GOOGLE_DRIVE / "processed"

RAW_DIR_EN = GOOGLE_DRIVE_EN / "raw"
PROCESSED_DIR_EN = GOOGLE_DRIVE_EN / "processed"

YEARS = range(2010, 2011)
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
