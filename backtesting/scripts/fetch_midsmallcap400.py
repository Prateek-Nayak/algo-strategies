"""
Fetch Nifty MidSmallcap 400 constituent symbols from NSE and save as JSON.

Usage:
    python backtesting/scripts/fetch_midsmallcap400.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import requests

NSE_BASE = "https://www.nseindia.com"
NSE_API = f"{NSE_BASE}/api/equity-stockIndices"
INDEX_NAME = "NIFTY MIDSMALLCAP 400"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "nifty_midsmallcap400.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nifty_500" / "partitioned_data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{NSE_BASE}/market-data/live-equity-market",
}


def fetch_constituents() -> list[str]:
    session = requests.Session()
    session.headers.update(HEADERS)

    # Seed cookies by visiting the main page
    session.get(NSE_BASE, timeout=10)

    resp = session.get(
        NSE_API,
        params={"index": INDEX_NAME},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    symbols = sorted(
        entry["symbol"]
        for entry in data.get("data", [])
        if entry.get("symbol") and entry["symbol"] != INDEX_NAME
    )
    return symbols


def main() -> None:
    print(f"Fetching {INDEX_NAME} constituents from NSE...")
    symbols = fetch_constituents()
    print(f"  Found {len(symbols)} symbols")

    payload = {
        "index": INDEX_NAME,
        "fetched_at": date.today().isoformat(),
        "symbols": symbols,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved to {OUTPUT_PATH}")

    # Intersection report
    if DATA_DIR.exists():
        available = {d.name.split("=")[1] for d in DATA_DIR.iterdir() if d.is_dir() and "=" in d.name}
        matched = sorted(set(symbols) & available)
        missing = sorted(set(symbols) - available)
        print(f"\n  Available in data: {len(matched)} / {len(symbols)}")
        if missing:
            print(f"  Missing from data ({len(missing)}): {', '.join(missing[:20])}{'...' if len(missing) > 20 else ''}")
    else:
        print(f"\n  Data dir not found: {DATA_DIR}")


if __name__ == "__main__":
    main()
