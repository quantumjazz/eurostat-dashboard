# CLAUDE.md

## Project Overview

Eurostat Dashboard — an economic data visualization tool that fetches EU statistical data from the Eurostat API, and data on non-EU OECD countries from the OECD API. Then it caches the data locally, and renders interactive charts comparing Bulgaria against the EU27 average. Will cover all main economic sectors. 

## Architecture

- **Backend**: Python FastAPI app (`backend/main.py`) serving a REST API. Uses httpx for async HTTP requests to Eurostat, pyjstat + pandas for data transformation, and SQLite for 24-hour caching.
- **Frontend**: Single-page vanilla HTML/CSS/JS app (`frontend/index.html`) using ECharts for chart rendering. No build step.
- **Deployment**: Backend on Railway, frontend on Vercel. Custom domain: visiometrica.com.

## Tech Stack

| Layer    | Technology                          |
|----------|-------------------------------------|
| Backend  | Python 3.13, FastAPI, Uvicorn, httpx, pandas, pyjstat |
| Frontend | Vanilla JS, ECharts 5.4.3 (CDN)    |
| Cache    | SQLite (`cache/eurostat_cache.db`)  |
| Deploy   | Railway (backend), Vercel (frontend)|

## Local Development

```bash
# Backend (terminal 1)
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload --port 8000

# Frontend (terminal 2)
cd frontend && python3 -m http.server 3000
```

Open http://localhost:3000. The frontend auto-detects localhost and points to `http://localhost:8000/api/v1`.

## API Endpoints

- `GET /api/v1/health` — returns `{"status": "ok"}`
- `GET /api/v1/data` — Eurostat data endpoint
  - Required params: `dataset`, `geo` (comma-separated), `time` (e.g. `2010-2024`)
  - Optional params: `unit`, `na_item`, `nace_r2`
  - Returns `{"data": [...], "cached": true|false}`
- `GET /api/v1/oecd` — OECD Economic Outlook data endpoint
  - Required params: `measures` (comma-separated, e.g. `GDPVD_CAP,UNR,GGFLQ`), `geo` (comma-separated, e.g. `USA,GBR,BGR`), `time` (e.g. `2020-2025`)
  - Returns `{"data": [...], "cached": true|false}` where each record has `geo`, `geo_name`, `measure`, `measure_name`, `time`, `value`
- `GET /api/v1/sdi` — SDI-EU (Sector Development Index) endpoint
  - Required params: `time` (e.g. `2015-2023`)
  - Computes three pillars: Labour Productivity, Compensation per Employee, Investment Intensity (each as BG/EU × 100)
  - Composite SDI = geometric mean of three pillars. EU27 = 100 baseline.
  - Returns `{"data": [...], "cached": true|false}` where each record has `nace`, `nace_label`, `time`, `lp`, `comp`, `inv`, `sdi`
  - Data sources: `nama_10_a10` (GVA, Compensation), `nama_10_a10_e` (Employment), `nama_10_a64_p5` (GFCF)
  - EU27 GFCF aggregate is computed by summing all 27 member states (not available as direct aggregate)

## Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app: API routes, Eurostat fetching, caching, data transformation |
| `backend/requirements.txt` | Python dependencies |
| `frontend/index.html` | Complete SPA: all charts, styles, and JS logic |
| `backend/Procfile` | Railway start command |
| `backend/railway.json` | Railway platform config |

## Datasets Used

### Eurostat
- `sdg_08_10` — GDP per capita (EUR, chain-linked 2020)
- `tipsna70` — Real labour productivity (Index 2015=100)
- `nama_10_a10` — Gross value added (B1G) and Compensation of employees (D1) by NACE sector
- `nama_10_a10_e` — Employment by NACE sector (EMP_DC, thousands of persons)
- `nama_10_a64_p5` — Gross fixed capital formation (P51G) by NACE sector and asset type (filter `asset10=N11G` for total)

### OECD (Economic Outlook, `DSD_EO@DF_EO`)
- `GDPVD_CAP` — GDP per capita (PPP, USD)
- `UNR` — Unemployment rate (% of labour force)
- `GGFLQ` — Government gross debt (% of GDP)
- `CBGDPR` — Current account balance (% of GDP)
- Country codes: `BGR` (Bulgaria), `USA`, `GBR`, `JPN`, `KOR`, `CAN`, `AUS`, `OECD` (aggregate)
- API base: `https://sdmx.oecd.org/public/rest/data/` (public, no auth, 60 req/hour limit)

## CORS

Allowed origins are configured in `backend/main.py`:
- `http://localhost:3000` (dev)
- `https://visiometrica.com`, `https://www.visiometrica.com` (prod)
- `https://eurostat-dashboard.vercel.app` (Vercel)

## Caching

SQLite-based with 24-hour TTL. Cache keys combine dataset, geo, time, unit, na_item, and nace_r2 parameters. Cache DB lives at `cache/eurostat_cache.db` and is gitignored.

## Frontend Structure

The SPA has three pages, toggled by navigation tabs:
- **Home** — KPI cards (GDP, unemployment, debt for Bulgaria vs OECD avg), horizontal bar chart comparing countries, GDP trend line chart. Data from OECD API.
- **Overview** — Sector-level bar charts comparing Bulgaria vs EU27 (GVA by NACE sector as % of GDP). Data from Eurostat API.
- **SDI-EU** — Sector Development Index with three interactive panels:
  - Panel 1: SDI Overview (horizontal bar chart, color-coded by band)
  - Panel 2: Pillar Decomposition (heatmap: 10 sectors × 3 pillars)
  - Panel 3: Convergence Over Time (multi-line trend chart)
  - Cross-panel interaction: clicking a sector highlights it across all panels.

The Overview and SDI-EU pages are lazy-loaded: they only fetch data when first visited.

## Testing

No automated test suite. Test manually via browser or curl:
```bash
curl http://localhost:8000/api/v1/health
curl "http://localhost:8000/api/v1/data?dataset=sdg_08_10&geo=BG,EU27_2020&time=2010-2024"
curl "http://localhost:8000/api/v1/oecd?measures=GDPVD_CAP,UNR,GGFLQ&geo=BGR,USA,GBR,JPN,KOR,OECD&time=2020-2025"
curl "http://localhost:8000/api/v1/sdi?time=2020-2023"
```

## Common Tasks

- **Add a new Eurostat chart**: Add a new `<div>` in the Charts page section of `frontend/index.html`, then add a render function using `fetchData()`.
- **Add a new OECD indicator**: Use the existing `/api/v1/oecd` endpoint with a new measure code from the frontend via `fetchOECDData()`.
- **Add a new Eurostat dataset**: No backend changes needed if the existing `/api/v1/data` params cover it. Just call the endpoint with the new dataset code.
- **Add a new OECD dataset**: Add a new endpoint in `backend/main.py` following the `fetch_oecd_data()` / `transform_oecd_csv()` pattern. Change the agency and dataflow constants as needed.
- **Change allowed CORS origins**: Edit the `origins` list in `backend/main.py`.
- **Clear cache**: Delete `cache/eurostat_cache.db`. It will be recreated on next request.
