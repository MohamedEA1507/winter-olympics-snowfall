"""
Country code reference tables used across the pipeline.

Three different code systems are in use:
  ISO3       — ISO 3166-1 alpha-3 (e.g. CHE, NLD). Used by World Bank and snowfall data.
  NOC        — National Olympic Committee codes (e.g. SUI, NED). Used by the Olympics/Kaggle data.
  regionmask — Short abbreviations from the Natural Earth dataset (e.g. "A", "CH", "SLO").

These three systems overlap but are not the same. The dicts below handle the translation between them.
"""

# Canonical NOC → ISO3 crosswalk used across the full pipeline.
# Replaces the old WIKI_NOC_TO_ISO3 (which only covered a subset of nations).
#
# Olympic datasets use IOC National Olympic Committee (NOC) codes, which differ
# from ISO 3166-1 alpha-3 codes used by the World Bank and ERA5.
# Two categories of mapping:
#   1. Simple substitutions: IOC chose a different 3-letter code than ISO
#      (e.g. GER→DEU, NED→NLD, SUI→CHE)
#   2. Defunct states: countries that no longer exist, mapped to modern successors.
#      Each decision is documented:
#        URS (Soviet Union) → RUS: Russia is the recognised successor state.
#        EUN (Unified Team 1992) → RUS: former Soviet republics competing together
#            one last time; attributed to RUS for continuity with URS records.
#        GDR (East Germany) → DEU: merged into unified Germany in 1990.
#        FRG (West Germany) → DEU: same reasoning as GDR.
#        TCH (Czechoslovakia) → CZE: Czech Republic treated as primary successor
#            for Winter Olympics purposes (Slovakia competed separately from 1994).
#        YUG (Yugoslavia) → SRB: Serbia is the recognised successor state.
#        SCG (Serbia-Montenegro 2006) → SRB: Montenegro declared independence 2006.
#
# Special cases:
#   OAR = "Olympic Athletes from Russia" (2018, banned under own flag) → RUS
#   ROC = "Russian Olympic Committee"    (2022, banned under own flag) → RUS
#   AIN = "Individual Neutral Athletes"  (2022, Belarusian athletes)   → BLR
NOC_TO_ISO3 = {
    # --- Simple code differences (same country, different code system) ---
    "GER": "DEU",   # Germany: IOC uses GER, ISO uses DEU
    "NED": "NLD",   # Netherlands: IOC uses NED, ISO uses NLD
    "SUI": "CHE",   # Switzerland: IOC uses SUI, ISO uses CHE
    "DEN": "DNK",   # Denmark: IOC uses DEN, ISO uses DNK
    "POR": "PRT",   # Portugal
    "SLO": "SVN",   # Slovenia: IOC uses SLO, ISO uses SVN
    "CRO": "HRV",   # Croatia
    "LAT": "LVA",   # Latvia
    "MGL": "MNG",   # Mongolia
    "PHI": "PHL",   # Philippines
    "TRI": "TTO",   # Trinidad and Tobago
    "ZIM": "ZWE",   # Zimbabwe
    "IRI": "IRN",   # Iran
    "MAS": "MYS",   # Malaysia
    "UAE": "ARE",   # United Arab Emirates
    "VIE": "VNM",   # Vietnam
    "TPE": "TWN",   # Chinese Taipei (Taiwan) — politically sensitive;
                    # IOC uses TPE, ISO uses TWN
    "GRE": "GRC",   # Greece
    "BUL": "BGR",   # Bulgaria
    "HAI": "HTI",   # Haiti
    "CHI": "CHL",   # Chile: IOC uses CHI, ISO uses CHL
    "RSA": "ZAF",   # South Africa
    "ALG": "DZA",   # Algeria
    "PAR": "PRY",   # Paraguay
    "URU": "URY",   # Uruguay
    "CRC": "CRI",   # Costa Rica
    "GUA": "GTM",   # Guatemala
    "HON": "HND",   # Honduras
    "ESA": "SLV",   # El Salvador
    "NCA": "NIC",   # Nicaragua
    "BAR": "BRB",   # Barbados
    "SKN": "KNA",   # Saint Kitts and Nevis
    "TGA": "TON",   # Tonga
    "ANT": "ATG",   # Antigua and Barbuda
    "ISV": "VIR",   # US Virgin Islands
    "PUR": "PRI",   # Puerto Rico (US territory, competes separately)
    "GUM": "GUM",   # Guam (no ISO3; kept as-is)
    "ASA": "ASM",   # American Samoa
    "HKG": "HKG",   # Hong Kong (special administrative region)
    "ANG": "AGO",   # Angola
    # --- Defunct states → modern successors ---
    "URS": "RUS",   # Soviet Union → Russia
    "EUN": "RUS",   # Unified Team 1992 → Russia
    "OAR": "RUS",   # Olympic Athletes from Russia (2018) → Russia
    "ROC": "RUS",   # Russian Olympic Committee (2022) → Russia
    "AIN": "BLR",   # Individual Neutral Athletes (2022, Belarusian) → Belarus
    "GDR": "DEU",   # East Germany → unified Germany
    "FRG": "DEU",   # West Germany → unified Germany
    "TCH": "CZE",   # Czechoslovakia → Czech Republic
    "YUG": "SRB",   # Yugoslavia → Serbia
    "SCG": "SRB",   # Serbia-Montenegro → Serbia
    "NGR": "NGA",   # Nigeria: IOC uses NGR, ISO uses NGA
    "BER": "BMU",   # Bermuda: IOC uses BER, ISO uses BMU
    "LIB": "LBN",   # Lebanon: IOC uses LIB, ISO uses LBN
    "MON": "MCO",   # Monaco: IOC uses MON, ISO uses MCO
    "NEP": "NPL",   # Nepal: IOC uses NEP, ISO uses NPL
    "MAD": "MDG",   # Madagascar: IOC uses MAD, ISO uses MDG
    "KSA": "SAU",   # Saudi Arabia: IOC uses KSA, ISO uses SAU
    "IVB": "VGB",   # British Virgin Islands: IOC uses IVB, ISO uses VGB
    "CAY": "CYM",   # Cayman Islands: IOC uses CAY, ISO uses CYM
    "KOS": "XKX",   # Kosovo: IOC uses KOS, ISO uses XKX (unrecognised state)
    "FIJ": "FJI",   # Fiji: IOC uses FIJ, ISO uses FJI
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