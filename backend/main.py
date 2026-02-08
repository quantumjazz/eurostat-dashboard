from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import sqlite3
import json
import math
import time
from io import StringIO
from pathlib import Path
from typing import Optional
from pyjstat import pyjstat
import pandas as pd

app = FastAPI(title="Eurostat Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://visiometrica.com",
        "https://www.visiometrica.com",
        "https://eurostat-dashboard.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DB = CACHE_DIR / "eurostat_cache.db"
CACHE_TTL = 3600 * 24  # 24 hours

EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

# OECD SDMX REST API
OECD_BASE_URL = "https://sdmx.oecd.org/public/rest/data"
OECD_AGENCY = "OECD.ECO.MAD"
OECD_DATAFLOW = "DSD_EO@DF_EO"


def init_cache():
    """Initialize SQLite cache database."""
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            data TEXT,
            timestamp INTEGER
        )
    """)
    conn.commit()
    conn.close()


def get_cached(key: str) -> Optional[dict]:
    """Get data from cache if not expired."""
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.execute(
        "SELECT data, timestamp FROM cache WHERE key = ?", (key,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        data, timestamp = row
        if time.time() - timestamp < CACHE_TTL:
            return json.loads(data)
    return None


def set_cached(key: str, data: dict):
    """Store data in cache."""
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, data, timestamp) VALUES (?, ?, ?)",
        (key, json.dumps(data), int(time.time()))
    )
    conn.commit()
    conn.close()


async def fetch_eurostat_data(dataset: str, geo: list[str], extra_params: Optional[dict] = None) -> dict:
    """Fetch data from Eurostat JSON-stat API."""
    # Build query parameters
    params = {
        "format": "JSON",
        "lang": "en",
        "geo": geo,
    }

    # Add any extra filter parameters (unit, na_item, nace_r2, etc.)
    if extra_params:
        params.update(extra_params)

    # Build URL
    url = f"{EUROSTAT_BASE_URL}/{dataset}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Eurostat API error: {response.text}"
            )

        return response.json()


def transform_jsonstat_to_tidy(jsonstat_data: dict, geo_filter: list[str], time_range: tuple[int, int], include_nace: bool = False) -> list[dict]:
    """Transform JSON-stat format to tidy row-oriented JSON."""
    # Use pyjstat to convert to DataFrame
    df = pyjstat.from_json_stat(jsonstat_data)[0]

    # Normalize column names - pyjstat uses full labels, we want short names
    col_mapping = {}
    unit_col = None
    nace_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'geopolitical' in col_lower or col_lower == 'geo':
            col_mapping[col] = 'geo'
        elif col_lower == 'time' or col_lower == 'time period':
            col_mapping[col] = 'time'
        elif col_lower == 'value':
            col_mapping[col] = 'value'
        # Check NACE before unit (NACE columns contain "Community" which has "unit")
        elif 'nace' in col_lower or 'economic activities' in col_lower:
            nace_col = col
            col_mapping[col] = 'nace'
        elif 'unit' in col_lower:
            unit_col = col

    df = df.rename(columns=col_mapping)

    # Filter by unit of measure if multiple exist (keep absolute values, not percentages)
    if unit_col and unit_col in df.columns:
        units = df[unit_col].unique()
        if len(units) > 1:
            # Prefer absolute values over percentage changes
            for unit in units:
                if 'percentage' not in unit.lower() and 'change' not in unit.lower():
                    df = df[df[unit_col] == unit]
                    break

    # Extract geo codes from the dimension metadata
    geo_dimension = jsonstat_data.get('dimension', {}).get('geo', {})
    geo_categories = geo_dimension.get('category', {})
    geo_labels = geo_categories.get('label', {})
    # Create reverse mapping: label -> code
    label_to_code = {v: k for k, v in geo_labels.items()}

    # Map geo labels back to codes if needed
    if 'geo' in df.columns:
        df['geo'] = df['geo'].apply(lambda x: label_to_code.get(x, x))

    # Extract NACE codes from dimension metadata if present
    if 'nace' in df.columns:
        nace_dimension = jsonstat_data.get('dimension', {}).get('nace_r2', {})
        nace_categories = nace_dimension.get('category', {})
        nace_labels = nace_categories.get('label', {})
        nace_label_to_code = {v: k for k, v in nace_labels.items()}
        df['nace'] = df['nace'].apply(lambda x: nace_label_to_code.get(x, x))

    # Filter by geo codes
    if geo_filter and 'geo' in df.columns:
        df = df[df['geo'].isin(geo_filter)]

    # Filter by time range
    if 'time' in df.columns:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df = df[(df['time'] >= time_range[0]) & (df['time'] <= time_range[1])]

    # Keep only essential columns
    essential_cols = ['geo', 'time', 'value']
    if include_nace and 'nace' in df.columns:
        essential_cols = ['geo', 'nace', 'time', 'value']
    available_cols = [c for c in essential_cols if c in df.columns]
    df = df[available_cols]

    # Sort by geo and time
    sort_cols = [c for c in ['geo', 'nace', 'time'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Convert to list of dicts
    return df.to_dict(orient='records')


@app.on_event("startup")
async def startup():
    init_cache()


@app.get("/api/v1/data")
async def get_data(
    dataset: str = Query(..., description="Eurostat dataset code"),
    geo: str = Query(..., description="Comma-separated geo codes"),
    time: str = Query(..., description="Time range as start-end (e.g., 2010-2023)"),
    unit: Optional[str] = Query(None, description="Unit filter (e.g., PC_GDP, I15)"),
    na_item: Optional[str] = Query(None, description="National accounts item (e.g., B1G for GVA)"),
    nace_r2: Optional[str] = Query(None, description="NACE sectors, comma-separated"),
):
    """
    Fetch and transform Eurostat data.

    Examples:
    - /api/v1/data?dataset=sdg_08_10&geo=BG,EU27_2020&time=2010-2024
    - /api/v1/data?dataset=tipsna70&geo=BG,EU27_2020&time=2010-2024&unit=I15
    - /api/v1/data?dataset=nama_10_a10&geo=BG&time=2010-2024&unit=PC_GDP&na_item=B1G
    """
    # Parse parameters
    geo_list = [g.strip() for g in geo.split(",")]
    time_parts = time.split("-")
    if len(time_parts) != 2:
        raise HTTPException(400, "Time must be in format start-end (e.g., 2010-2023)")
    time_range = (int(time_parts[0]), int(time_parts[1]))

    # Build extra params for Eurostat API
    extra_params = {}
    if unit:
        extra_params["unit"] = unit
    if na_item:
        extra_params["na_item"] = na_item
    if nace_r2:
        extra_params["nace_r2"] = [n.strip() for n in nace_r2.split(",")]

    # Check cache
    cache_key = f"{dataset}:{geo}:{time}:{unit}:{na_item}:{nace_r2}"
    cached = get_cached(cache_key)
    if cached:
        return {"data": cached, "cached": True}

    # Fetch from Eurostat
    raw_data = await fetch_eurostat_data(dataset, geo_list, extra_params if extra_params else None)

    # Transform to tidy format
    include_nace = nace_r2 is not None or dataset == "nama_10_a10"
    tidy_data = transform_jsonstat_to_tidy(raw_data, geo_list, time_range, include_nace=include_nace)

    # Cache the result
    set_cached(cache_key, tidy_data)

    return {"data": tidy_data, "cached": False}


async def fetch_oecd_data(countries: list[str], measures: list[str], start_year: int, end_year: int) -> str:
    """Fetch data from the OECD SDMX REST API in CSV format."""
    geo = "+".join(countries)
    measure = "+".join(measures)

    url = (
        f"{OECD_BASE_URL}/{OECD_AGENCY},{OECD_DATAFLOW},/"
        f"{geo}.{measure}.A"
        f"?startPeriod={start_year}"
        f"&endPeriod={end_year}"
        f"&dimensionAtObservation=AllDimensions"
        f"&format=csvfilewithlabels"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OECD API error: {response.text[:500]}"
            )

        return response.text


def transform_oecd_csv(csv_text: str) -> list[dict]:
    """Transform OECD CSV response to tidy row-oriented JSON."""
    df = pd.read_csv(StringIO(csv_text))

    # Keep only the columns we need
    keep_cols = ["REF_AREA", "Reference area", "MEASURE", "Measure", "TIME_PERIOD", "OBS_VALUE"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Rename to our standard format
    rename_map = {
        "REF_AREA": "geo",
        "Reference area": "geo_name",
        "MEASURE": "measure",
        "Measure": "measure_name",
        "TIME_PERIOD": "time",
        "OBS_VALUE": "value",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert types
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop rows with missing values
    df = df.dropna(subset=["value"])

    # Sort
    sort_cols = [c for c in ["geo", "measure", "time"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df.to_dict(orient="records")


@app.get("/api/v1/oecd")
async def get_oecd_data(
    measures: str = Query(..., description="Comma-separated OECD measure codes (e.g., GDPVD_CAP,UNR,GGFLQ)"),
    geo: str = Query(..., description="Comma-separated country codes (e.g., USA,GBR,JPN,KOR,BGR)"),
    time: str = Query(..., description="Time range as start-end (e.g., 2020-2025)"),
):
    """
    Fetch and transform OECD Economic Outlook data.

    Key measure codes:
    - GDPVD_CAP: GDP per capita (PPP, USD)
    - UNR: Unemployment rate (%)
    - CPI: Consumer Price Index
    - GGFLQ: Government gross debt (% of GDP)
    - CBGDPR: Current account balance (% of GDP)

    Example:
    - /api/v1/oecd?measures=GDPVD_CAP,UNR&geo=USA,GBR,JPN,KOR,BGR&time=2020-2025
    """
    # Parse parameters
    measure_list = [m.strip() for m in measures.split(",")]
    geo_list = [g.strip() for g in geo.split(",")]
    time_parts = time.split("-")
    if len(time_parts) != 2:
        raise HTTPException(400, "Time must be in format start-end (e.g., 2020-2025)")
    start_year = int(time_parts[0])
    end_year = int(time_parts[1])

    # Check cache
    cache_key = f"oecd:{measures}:{geo}:{time}"
    cached = get_cached(cache_key)
    if cached:
        return {"data": cached, "cached": True}

    # Fetch from OECD
    csv_text = await fetch_oecd_data(geo_list, measure_list, start_year, end_year)

    # Transform to tidy format
    tidy_data = transform_oecd_csv(csv_text)

    # Cache the result
    set_cached(cache_key, tidy_data)

    return {"data": tidy_data, "cached": False}


# ==================== SDI-EU (Sector Development Index) ====================

NACE_A10_SECTORS = ["A", "B-E", "F", "G-I", "J", "K", "L", "M_N", "O-Q", "R-U"]
NACE_LABELS = {
    "A": "Agriculture",
    "B-E": "Industry",
    "F": "Construction",
    "G-I": "Trade & Transport",
    "J": "ICT",
    "K": "Finance",
    "L": "Real Estate",
    "M_N": "Professional Svcs",
    "O-Q": "Public Admin",
    "R-U": "Arts & Other",
}

# All 27 EU member states (for computing EU27 aggregate where not directly available)
EU27_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "EL", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]


async def fetch_eurostat_for_sdi(
    dataset: str,
    geo: list[str],
    nace_r2: list[str],
    time_range: tuple[int, int],
    unit: str,
    na_item: Optional[str] = None,
    extra_filters: Optional[dict] = None,
) -> pd.DataFrame:
    """Fetch a single Eurostat dataset and return a tidy DataFrame with geo, nace, time, value."""
    extra_params = {"unit": unit, "nace_r2": nace_r2}
    if na_item:
        extra_params["na_item"] = na_item
    if extra_filters:
        extra_params.update(extra_filters)

    raw = await fetch_eurostat_data(dataset, geo, extra_params)
    rows = transform_jsonstat_to_tidy(raw, geo, time_range, include_nace=True)
    df = pd.DataFrame(rows)

    # For employment dataset, filter to EMP_DC if there are multiple na_item values
    # (the transform already drops the na_item column, but multiple might appear as separate rows)
    # We handle this by trusting the Eurostat filter — if na_item was passed, only matching rows return.

    if df.empty:
        return df

    # Ensure numeric types
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    return df


def compute_sdi(
    gva_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    emp_df: pd.DataFrame,
    gfcf_bg_df: pd.DataFrame,
    gfcf_eu_df: pd.DataFrame,
) -> list[dict]:
    """Compute SDI-EU index from raw data DataFrames.

    Returns list of dicts with: nace, nace_label, time, lp, comp, inv, sdi
    """
    results = []

    # Build lookup dicts: (geo, nace, time) -> value
    def build_lookup(df: pd.DataFrame) -> dict:
        lookup = {}
        for _, row in df.iterrows():
            key = (row["geo"], row["nace"], int(row["time"]))
            lookup[key] = row["value"]
        return lookup

    gva_lookup = build_lookup(gva_df)
    comp_lookup = build_lookup(comp_df)
    emp_lookup = build_lookup(emp_df)

    # GFCF BG lookup
    gfcf_bg_lookup = build_lookup(gfcf_bg_df)

    # GFCF EU27: aggregate from individual countries
    gfcf_eu_lookup = {}
    for _, row in gfcf_eu_df.iterrows():
        key = (row["nace"], int(row["time"]))
        geo = row["geo"]
        if geo == "BG":
            continue  # BG is already handled separately
        gfcf_eu_lookup.setdefault(key, 0.0)
        gfcf_eu_lookup[key] += row["value"]

    # Get all available years
    all_years = sorted(set(int(r["time"]) for _, r in gva_df.iterrows()))

    for nace in NACE_A10_SECTORS:
        for year in all_years:
            # Get values
            gva_bg = gva_lookup.get(("BG", nace, year))
            gva_eu = gva_lookup.get(("EU27_2020", nace, year))
            comp_bg = comp_lookup.get(("BG", nace, year))
            comp_eu = comp_lookup.get(("EU27_2020", nace, year))
            emp_bg = emp_lookup.get(("BG", nace, year))
            emp_eu = emp_lookup.get(("EU27_2020", nace, year))
            gfcf_bg = gfcf_bg_lookup.get(("BG", nace, year))
            gfcf_eu = gfcf_eu_lookup.get((nace, year))

            # Skip if any essential data is missing
            if not all([gva_bg, gva_eu, comp_bg, comp_eu, emp_bg, emp_eu]):
                continue

            # Compute Labour Productivity relative index
            lp_bg = gva_bg / emp_bg
            lp_eu = gva_eu / emp_eu
            lp_rel = (lp_bg / lp_eu) * 100 if lp_eu else None

            # Compute Compensation per Employee relative index
            comp_bg_val = comp_bg / emp_bg
            comp_eu_val = comp_eu / emp_eu
            comp_rel = (comp_bg_val / comp_eu_val) * 100 if comp_eu_val else None

            # Compute Investment Intensity relative index (if GFCF data available)
            inv_rel = None
            if gfcf_bg and gfcf_eu and gva_bg and gva_eu:
                inv_bg = gfcf_bg / gva_bg
                inv_eu = gfcf_eu / gva_eu
                inv_rel = (inv_bg / inv_eu) * 100 if inv_eu else None

            if lp_rel is None or comp_rel is None:
                continue

            # Compute composite SDI
            if inv_rel is not None and inv_rel > 0:
                sdi = (lp_rel * comp_rel * inv_rel) ** (1 / 3)
            else:
                # Fall back to geometric mean of 2 pillars if investment data unavailable
                sdi = (lp_rel * comp_rel) ** (1 / 2)

            results.append({
                "nace": nace,
                "nace_label": NACE_LABELS.get(nace, nace),
                "time": year,
                "lp": round(lp_rel, 1),
                "comp": round(comp_rel, 1),
                "inv": round(inv_rel, 1) if inv_rel is not None else None,
                "sdi": round(sdi, 1),
            })

    return results


@app.get("/api/v1/sdi")
async def get_sdi_data(
    time: str = Query(..., description="Time range as start-end (e.g., 2015-2023)"),
):
    """
    Compute and return the SDI-EU (Sector Development Index relative to EU).

    Three pillars:
    - Labour Productivity: (GVA/Employment)_BG / (GVA/Employment)_EU × 100
    - Compensation per Employee: (Compensation/Employment)_BG / (Compensation/Employment)_EU × 100
    - Investment Intensity: (GFCF/GVA)_BG / (GFCF/GVA)_EU × 100

    Composite SDI = geometric mean of three pillars. EU27 = 100.

    Example:
    - /api/v1/sdi?time=2015-2023
    """
    time_parts = time.split("-")
    if len(time_parts) != 2:
        raise HTTPException(400, "Time must be in format start-end (e.g., 2015-2023)")
    time_range = (int(time_parts[0]), int(time_parts[1]))

    # Check cache
    cache_key = f"sdi:{time}"
    cached = get_cached(cache_key)
    if cached:
        return {"data": cached, "cached": True}

    # Fetch all datasets concurrently
    naces = NACE_A10_SECTORS

    gva_task = fetch_eurostat_for_sdi(
        "nama_10_a10", ["BG", "EU27_2020"], naces, time_range,
        unit="CP_MEUR", na_item="B1G",
    )
    comp_task = fetch_eurostat_for_sdi(
        "nama_10_a10", ["BG", "EU27_2020"], naces, time_range,
        unit="CP_MEUR", na_item="D1",
    )
    emp_task = fetch_eurostat_for_sdi(
        "nama_10_a10_e", ["BG", "EU27_2020"], naces, time_range,
        unit="THS_PER", na_item="EMP_DC",
    )
    # GFCF: BG direct, EU27 from individual countries
    # Filter asset10=N11G (Total fixed assets) to avoid asset-type breakdown
    gfcf_bg_task = fetch_eurostat_for_sdi(
        "nama_10_a64_p5", ["BG"], naces, time_range,
        unit="CP_MEUR", na_item="P51G",
        extra_filters={"asset10": "N11G"},
    )
    gfcf_eu_task = fetch_eurostat_for_sdi(
        "nama_10_a64_p5", EU27_COUNTRIES, naces, time_range,
        unit="CP_MEUR", na_item="P51G",
        extra_filters={"asset10": "N11G"},
    )

    gva_df, comp_df, emp_df, gfcf_bg_df, gfcf_eu_df = await asyncio.gather(
        gva_task, comp_task, emp_task, gfcf_bg_task, gfcf_eu_task
    )

    # Compute SDI
    sdi_data = compute_sdi(gva_df, comp_df, emp_df, gfcf_bg_df, gfcf_eu_df)

    # Cache the result
    set_cached(cache_key, sdi_data)

    return {"data": sdi_data, "cached": False}


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
