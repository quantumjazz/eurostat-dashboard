# Eurostat Dashboard

A dashboard application that visualizes economic indicators from Eurostat, comparing Bulgaria with the EU27 average.

## Features

- **GDP per Capita** - Chain linked volumes (2020), euro per capita
- **Labour Productivity** - Real labour productivity per hour worked (Index, 2015=100)
- **GVA by Sector** - Gross value added breakdown by NACE sectors (% of GDP)

## Architecture

```
eurostat_dashboard/
├── backend/                 # FastAPI REST API (deployed to Railway)
│   ├── main.py             # API endpoints & data transformation
│   ├── requirements.txt
│   ├── Procfile            # Railway start command
│   └── railway.json        # Railway config
├── frontend/               # Static dashboard (deployed to Vercel)
│   └── index.html          # Single-page app with ECharts
├── cache/                  # SQLite cache (gitignored)
├── DEPLOYMENT.md           # Step-by-step deployment guide
└── README.md
```

## API

**Endpoint:** `GET /api/v1/data`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `dataset` | Yes | Eurostat dataset code (e.g., `sdg_08_10`) |
| `geo` | Yes | Comma-separated geo codes (e.g., `BG,EU27_2020`) |
| `time` | Yes | Time range as `start-end` (e.g., `2010-2024`) |
| `unit` | No | Unit filter (e.g., `PC_GDP`, `I15`) |
| `na_item` | No | National accounts item (e.g., `B1G` for GVA) |
| `nace_r2` | No | NACE sectors, comma-separated |

**Example:**
```
/api/v1/data?dataset=nama_10_a10&geo=BG&time=2010-2024&unit=PC_GDP&na_item=B1G&nace_r2=A,B-E,F
```

## Local Development

### Prerequisites
- Python 3.12+

### Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start backend (terminal 1)
cd backend && uvicorn main:app --reload --port 8000

# Start frontend (terminal 2)
cd frontend && python3 -m http.server 3000
```

Open http://localhost:3000

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete instructions to deploy:
- Backend → Railway
- Frontend → Vercel
- Custom domain configuration

## Datasets

| Code | Description | Source |
|------|-------------|--------|
| `sdg_08_10` | GDP per capita | [Eurostat](https://ec.europa.eu/eurostat/databrowser/view/sdg_08_10) |
| `tipsna70` | Labour productivity | [Eurostat](https://ec.europa.eu/eurostat/databrowser/view/tipsna70) |
| `nama_10_a10` | GVA by A*10 industry | [Eurostat](https://ec.europa.eu/eurostat/databrowser/view/nama_10_a10) |

## License

MIT
