from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sqlite3
import json
import time
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


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
