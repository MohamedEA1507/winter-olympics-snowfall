"""
Country code reference tables used across the pipeline.

Three different code systems are in use:
  ISO3       — ISO 3166-1 alpha-3 (e.g. CHE, NLD). Used by World Bank and snowfall data.
  NOC        — National Olympic Committee codes (e.g. SUI, NED). Used by the Olympics/Kaggle data.
  regionmask — Short abbreviations from the Natural Earth dataset (e.g. "A", "CH", "SLO").

These three systems overlap but are not the same. The dicts below handle the translation between them.
"""

# The regionmask library uses short abbreviations to identify countries. Many of these are 1–2 letter codes that don't match ISO 3166-1 alpha-3.
# This lookup table manually maps the ambiguous/non-standard ones to their correct ISO3 codes.
# Countries already using a standard 3-letter code (e.g. "DEU") don't need an entry here — they are handled automatically by the to_iso3() fallback logic in fetch.py.
REGIONMASK_TO_ISO3 = {
    "A":   "AUT",  "N":   "NOR",  "S":   "SWE",  "J":   "JPN",
    "CH":  "CHE",  "IS":  "ISL",  "IL":  "ISR",  "CL":  "CHL",
    "CA":  "CAN",  "US":  "USA",  "AL":  "ALB",  "ME":  "MNE",
    "KO":  "XKX",  "SLO": "SVN",  "SK":  "SVK",  "NM":  "MKD",
    "BG":  "BGR",  "TR":  "TUR",  "GE":  "GEO",  "GL":  "GRL",
    "AF":  "AFG",  "NP":  "NPL",  "BT":  "BTN",  "TJ":  "TJK",
    "KG":  "KGZ",  "TF":  "ATF",  "CZ":  "CZE",  "BiH": "BIH",
}

# Wikipedia uses IOC/NOC codes (e.g. "SUI" for Switzerland, "NED" for Netherlands) which differ from ISO 3166-1 alpha-3 codes (e.g. "CHE", "NLD").
# This mapping converts the NOC codes found in Wikipedia tables to standard ISO3.
# Special cases:
#   OAR = "Olympic Athletes from Russia" (2018, banned under own flag)
#   ROC = "Russian Olympic Committee" (2022, still banned under own flag)
#   AIN = "Individual Neutral Athletes" (2022, Belarusian athletes)
WIKI_NOC_TO_ISO3 = {
    "NOR": "NOR", "GER": "DEU", "CAN": "CAN", "USA": "USA", "NED": "NLD",
    "SWE": "SWE", "KOR": "KOR", "SUI": "CHE", "FRA": "FRA", "AUT": "AUT",
    "JPN": "JPN", "ITA": "ITA", "OAR": "RUS", "ROC": "RUS", "AIN": "RUS",
    "CZE": "CZE", "BLR": "BLR", "CHN": "CHN", "SVK": "SVK", "FIN": "FIN",
    "GBR": "GBR", "POL": "POL", "HUN": "HUN", "UKR": "UKR", "AUS": "AUS",
    "SLO": "SVN", "BEL": "BEL", "NZL": "NZL", "ESP": "ESP", "KAZ": "KAZ",
    "LAT": "LVA", "LIE": "LIE", "EST": "EST", "GEO": "GEO", "BUL": "BGR",
    "DEN": "DNK", "BRA": "BRA", "ARM": "ARM", "ROU": "ROU", "CRO": "HRV",
    "SRB": "SRB", "MEX": "MEX", "NIG": "NGA", "GRE": "GRC", "ISL": "ISL",
    "MKD": "MKD", "MNE": "MNE", "POR": "PRT", "RSA": "ZAF", "TUR": "TUR",
    "LTU": "LTU",
}

# Maps English country names (as they appear on Wikipedia) to NOC codes.
# NOC codes intentionally differ from ISO3 — e.g. Switzerland=SUI not CHE, Netherlands=NED not NLD.
# These match the codes used in the Kaggle dataset so both sources can be joined on noc_code.
ENGLISH_NAME_TO_NOC = {
    "Albania": "ALB", "Andorra": "AND", "Argentina": "ARG",
    "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Belarus": "BLR", "Belgium": "BEL",
    "Bermuda": "BER", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA", "Bulgaria": "BUL", "Canada": "CAN",
    "Chile": "CHI", "China": "CHN", "Chinese Taipei": "TPE",
    "Colombia": "COL", "Croatia": "CRO", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Czechia": "CZE", "Denmark": "DEN",
    "Ecuador": "ECU", "Eritrea": "ERI", "Estonia": "EST",
    "Finland": "FIN", "France": "FRA", "Georgia": "GEO",
    "Germany": "GER", "Ghana": "GHA", "Great Britain": "GBR",
    "Greece": "GRE", "Hong Kong": "HKG", "Hungary": "HUN",
    "Iceland": "ISL", "India": "IND", "Iran": "IRI",
    "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Jamaica": "JAM", "Japan": "JPN", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kosovo": "KOS", "Kyrgyzstan": "KGZ",
    "Latvia": "LAT", "Lebanon": "LIB", "Liechtenstein": "LIE",
    "Lithuania": "LTU", "Luxembourg": "LUX", "North Macedonia": "MKD",
    "Macedonia": "MKD", "Madagascar": "MAD", "Malaysia": "MAS",
    "Malta": "MLT", "Mexico": "MEX", "Moldova": "MDA",
    "Monaco": "MON", "Mongolia": "MGL", "Montenegro": "MNE",
    "Morocco": "MAR", "Netherlands": "NED", "New Zealand": "NZL",
    "Nigeria": "NGR", "North Korea": "PRK", "Norway": "NOR",
    "Olympic Athletes from Russia": "OAR", "ROC": "ROC",
    "Individual Neutral Athletes": "AIN", "Pakistan": "PAK",
    "Philippines": "PHI", "Poland": "POL", "Portugal": "POR",
    "Puerto Rico": "PUR", "Romania": "ROU", "Russia": "RUS",
    "San Marino": "SMR", "Serbia": "SRB", "Singapore": "SGP",
    "Slovakia": "SVK", "Slovenia": "SLO", "South Africa": "RSA",
    "South Korea": "KOR", "Korea": "KOR", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "SUI", "Thailand": "THA",
    "Timor-Leste": "TLS", "Togo": "TOG", "Tonga": "TGA",
    "Turkey": "TUR", "Ukraine": "UKR", "United States": "USA",
    "Uzbekistan": "UZB", "Tajikistan": "TJK", "Peru": "PER",
    "American Samoa": "ASA", "Virgin Islands": "ISV",
    "United Arab Emirates": "UAE", "Uruguay": "URU",
    "Venezuela": "VEN", "Zimbabwe": "ZIM", "Nepal": "NEP",
    "Paraguay": "PAR", "British Virgin Islands": "IVB",
    "Cayman Islands": "CAY", "Dominica": "DMA", "Indonesia": "INA",
    "Guatemala": "GUA", "Haiti": "HAI", "El Salvador": "ESA",
    "Trinidad and Tobago": "TTO", "Saudi Arabia": "KSA",
    "Benin": "BEN", "Guinea-Bissau": "GBS", "Senegal": "SEN",
    "Tunisia": "TUN", "Ethiopia": "ETH", "Tanzania": "TAN",
    "Uganda": "UGA", "Namibia": "NAM", "Cameroon": "CMR",
    "Egypt": "EGY", "Algeria": "ALG", "Fiji": "FIJ",
    "Samoa": "SAM", "Guam": "GUM",
}