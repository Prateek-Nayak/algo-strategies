"""
Intraday Short Reversal Strategy (Gap Short)

Short stocks that gap up more than their own 14-day average gap.
Trailing stop = gap %. Mandatory EOD exit.

Multi-stock equity strategy operating on Nifty 500 / MidSmallcap 400 universe.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

BACKTESTING_DIR = Path(__file__).resolve().parents[2]
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

from strategy_runtime import get_config, save_results  # noqa: E402

DATA_ROOT = BACKTESTING_DIR / "data"

DEFAULT_CONFIG = {
    "DATE_FROM": "2025-03-01",
    "DATE_TO": "2026-03-12",
    "STARTING_CAPITAL": 1000000,
    "POSITION_SIZE_PCT": 10.0,
    "MIN_SHARES": 10,
    "MIN_GAP_PCT": 0.50,
    "MIN_AVG_VOLUME": 100000,
    "GAP_LOOKBACK_DAYS": 14,
    "ENTRY_TIME": "09:15",
    "EOD_EXIT_TIME": "14:55",
    "SCAN_ENTRY_BAR": 1,
    "MAX_POSITIONS": 10,
    "BROKERAGE_PER_TRADE": 40,
    "SLIPPAGE_PCT": 0.05,
    "UNIVERSE_FILE": "nifty_midsmallcap400.json",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_date(text: str):
    return datetime.strptime(text, "%Y-%m-%d").date()


def parse_time(text: str):
    return datetime.strptime(text, "%H:%M").time()


def load_universe(cfg: dict) -> list[str]:
    """Return list of ticker symbols to scan."""
    universe_file = cfg.get("UNIVERSE_FILE", "nifty_midsmallcap400.json")
    data_dir = DATA_ROOT / "nifty_500" / "partitioned_data"

    if universe_file == "all":
        return sorted(
            d.name.split("=")[1]
            for d in data_dir.iterdir()
            if d.is_dir() and "=" in d.name
        )

    path = DATA_ROOT / universe_file
    if not path.exists():
        print(f"WARNING: Universe file {path} not found, using all available tickers")
        return sorted(
            d.name.split("=")[1]
            for d in data_dir.iterdir()
            if d.is_dir() and "=" in d.name
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    symbols = payload.get("symbols", [])
    # Intersect with available data
    available = {
        d.name.split("=")[1]
        for d in data_dir.iterdir()
        if d.is_dir() and "=" in d.name
    }
    return sorted(set(symbols) & available)


def load_ticker_1min(ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Load 1-min parquet data for a single ticker, filtered to date range."""
    ticker_dir = DATA_ROOT / "nifty_500" / "partitioned_data" / f"ticker={ticker}"
    if not ticker_dir.exists():
        return pd.DataFrame()

    from_year = int(date_from[:4])
    to_year = int(date_to[:4])

    frames = []
    for pf in sorted(ticker_dir.glob("*.parquet")):
        year = int(pf.stem)
        if year < from_year or year > to_year:
            continue
        frames.append(pd.read_parquet(pf))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["date"])
    df["trade_date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    return df.sort_values("datetime").reset_index(drop=True)


def load_ticker_day(ticker: str, trade_date, ticker_dir: Path) -> pd.DataFrame:
    """Load 1-min bars for a single ticker for a single date (lazy)."""
    year = trade_date.year
    pf = ticker_dir / f"{year}.parquet"
    if not pf.exists():
        return pd.DataFrame()

    df = pd.read_parquet(pf)
    df["datetime"] = pd.to_datetime(df["date"])
    df["trade_date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    df = df[df["trade_date"] == trade_date].sort_values("datetime").reset_index(drop=True)
    return df


# ── Phase 1: Build daily summaries ──────────────────────────────────────────


def build_daily_summaries(
    tickers: list[str], date_from: str, date_to: str
) -> dict[str, pd.DataFrame]:
    """
    For each ticker, build a DataFrame with one row per trading day:
    [date, open, close, high, low, volume]
    """
    from_dt = parse_date(date_from)
    to_dt = parse_date(date_to)
    summaries: dict[str, pd.DataFrame] = {}

    market_open = time(9, 15)

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Loading daily summaries: {i + 1}/{len(tickers)}")

        df = load_ticker_1min(ticker, date_from, date_to)
        if df.empty:
            continue

        # Filter to date range
        df = df[(df["trade_date"] >= from_dt) & (df["trade_date"] <= to_dt)]
        if df.empty:
            continue

        daily_rows = []
        for d, grp in df.groupby("trade_date"):
            grp = grp.sort_values("time")
            # Open = first bar's open (should be 09:15)
            first_bar = grp.iloc[0]
            last_bar = grp.iloc[-1]
            daily_rows.append({
                "date": d,
                "open": float(first_bar["open"]),
                "close": float(last_bar["close"]),
                "high": float(grp["high"].max()),
                "low": float(grp["low"].min()),
                "volume": int(grp["volume"].sum()),
            })

        if daily_rows:
            summary = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
            summaries[ticker] = summary

    return summaries


# ── Phase 2: Compute gaps and rolling averages ──────────────────────────────


def compute_gap_signals(
    summaries: dict[str, pd.DataFrame], lookback: int
) -> dict[str, pd.DataFrame]:
    """
    Add gap_pct, avg_gap, avg_volume columns to each ticker's daily summary.
    """
    enriched: dict[str, pd.DataFrame] = {}

    for ticker, df in summaries.items():
        if len(df) < lookback + 1:
            continue

        df = df.copy()
        df["prev_close"] = df["close"].shift(1)
        df["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"] * 100
        df["avg_gap"] = df["gap_pct"].rolling(lookback, min_periods=lookback).mean()
        df["avg_volume"] = df["volume"].rolling(lookback, min_periods=lookback).mean()

        # Drop warmup rows
        df = df.dropna(subset=["avg_gap", "avg_volume"]).reset_index(drop=True)
        if not df.empty:
            enriched[ticker] = df

    return enriched


# ── Phase 3: Day-by-day simulation ──────────────────────────────────────────


def run_simulation(
    enriched: dict[str, pd.DataFrame], cfg: dict
) -> pd.DataFrame:
    """Walk through each trading day, scanning for gaps and managing positions."""

    starting_capital = float(cfg["STARTING_CAPITAL"])
    position_pct = float(cfg["POSITION_SIZE_PCT"])
    min_shares = int(cfg["MIN_SHARES"])
    min_gap = float(cfg["MIN_GAP_PCT"])
    min_avg_vol = float(cfg["MIN_AVG_VOLUME"])
    max_positions = int(cfg["MAX_POSITIONS"])
    brokerage = float(cfg["BROKERAGE_PER_TRADE"])
    slippage_pct = float(cfg["SLIPPAGE_PCT"]) / 100
    eod_exit_t = parse_time(cfg["EOD_EXIT_TIME"])
    entry_bar_offset = int(cfg.get("SCAN_ENTRY_BAR", 1))
    date_from = parse_date(cfg["DATE_FROM"])
    date_to = parse_date(cfg["DATE_TO"])

    # Collect all trading dates across all tickers
    all_dates: set = set()
    # Build date-indexed lookups
    ticker_date_idx: dict[str, dict] = {}  # ticker -> {date -> row_dict}
    for ticker, df in enriched.items():
        idx = {}
        for _, row in df.iterrows():
            d = row["date"]
            if d >= date_from and d <= date_to:
                all_dates.add(d)
                idx[d] = row.to_dict()
        ticker_date_idx[ticker] = idx

    trading_dates = sorted(all_dates)
    portfolio_value = starting_capital
    all_trades: list[dict] = []

    data_base = DATA_ROOT / "nifty_500" / "partitioned_data"

    for day_i, trade_date in enumerate(trading_dates):
        if (day_i + 1) % 50 == 0:
            print(f"  Simulating day {day_i + 1}/{len(trading_dates)}: {trade_date}")

        # Step A: Screen candidates
        candidates = []
        for ticker, date_idx in ticker_date_idx.items():
            row = date_idx.get(trade_date)
            if row is None:
                continue
            gap = row["gap_pct"]
            avg_gap = row["avg_gap"]
            avg_vol = row["avg_volume"]

            if gap <= 0:
                continue
            if gap < min_gap:
                continue
            if gap <= avg_gap:
                continue
            if avg_vol < min_avg_vol:
                continue

            candidates.append({
                "ticker": ticker,
                "gap_pct": gap,
                "avg_gap_14d": avg_gap,
                "daily_open": row["open"],
                "prev_close": row["prev_close"],
            })

        if not candidates:
            continue

        # Sort by strongest gap first, cap at MAX_POSITIONS
        candidates.sort(key=lambda c: c["gap_pct"], reverse=True)
        candidates = candidates[:max_positions]

        # Step B: Size positions and load 1-min data for each
        entered_today: set = set()

        for cand in candidates:
            ticker = cand["ticker"]
            if ticker in entered_today:
                continue

            gap_pct = cand["gap_pct"]
            daily_open = cand["daily_open"]

            # Entry price with slippage (shorting, so we get a worse fill = lower price)
            entry_price = daily_open * (1 - slippage_pct)

            # Position sizing: fixed capital allocation
            target_amount = starting_capital * position_pct / 100
            shares = max(min_shares, math.floor(target_amount / entry_price))

            # Initial SL = entry * (1 + gap%)
            sl_pct = gap_pct / 100
            initial_sl = entry_price * (1 + sl_pct)

            # Load 1-min bars for this ticker on this date
            ticker_dir = data_base / f"ticker={ticker}"
            day_bars = load_ticker_day(ticker, trade_date, ticker_dir)
            if day_bars.empty:
                continue

            # Entry on the bar after the open (bar index = entry_bar_offset)
            if len(day_bars) <= entry_bar_offset:
                continue

            entry_bar = day_bars.iloc[entry_bar_offset]
            entry_time_val = entry_bar["time"]
            # Use the bar's open as the actual fill price (with slippage)
            entry_price = float(entry_bar["open"]) * (1 - slippage_pct)
            if entry_price <= 0:
                continue

            # Recalculate SL with actual entry price
            initial_sl = entry_price * (1 + sl_pct)

            # Walk forward through remaining bars
            post_entry = day_bars.iloc[entry_bar_offset + 1:]
            lowest_since_entry = entry_price
            active_sl = initial_sl
            exit_price = None
            exit_time_val = None
            exit_reason = "EOD"
            trail_log: list[dict] = []

            for _, bar in post_entry.iterrows():
                bar_time = bar["time"]
                bar_high = float(bar["high"])
                bar_low = float(bar["low"])
                bar_close = float(bar["close"])
                prev_sl = active_sl

                # Update lowest price since entry
                if bar_low < lowest_since_entry:
                    lowest_since_entry = bar_low
                    new_trail = lowest_since_entry * (1 + sl_pct)
                    if new_trail < active_sl:
                        active_sl = new_trail  # tighten only

                # Record trail log entry
                trail_log.append({
                    "t": str(bar_time)[:5],
                    "h": round(bar_high, 2),
                    "l": round(bar_low, 2),
                    "c": round(bar_close, 2),
                    "sl": round(active_sl, 2),
                    "low_wm": round(lowest_since_entry, 2),
                    "tightened": active_sl < prev_sl,
                })

                # Check trailing SL hit (short position: price rises to SL)
                if bar_high >= active_sl:
                    trail_log[-1]["action"] = "SL_HIT"
                    exit_price = active_sl * (1 + slippage_pct)  # slippage on buy-to-cover
                    exit_time_val = bar_time
                    exit_reason = "TRAILING_SL"
                    break

                # Check EOD exit
                if bar_time >= eod_exit_t:
                    trail_log[-1]["action"] = "EOD_EXIT"
                    exit_price = bar_close * (1 + slippage_pct)
                    exit_time_val = bar_time
                    exit_reason = "EOD"
                    break

            # If we never hit SL or EOD (shouldn't happen, but safety)
            if exit_price is None:
                last_bar = day_bars.iloc[-1]
                exit_price = float(last_bar["close"]) * (1 + slippage_pct)
                exit_time_val = last_bar["time"]
                exit_reason = "EOD"

            # PnL for short: entry - exit (per share)
            pnl_per_share = entry_price - exit_price
            gross_pnl = pnl_per_share * shares
            net_pnl = gross_pnl - (brokerage * 2)  # brokerage both sides
            portfolio_value += net_pnl
            entered_today.add(ticker)

            all_trades.append({
                "date": trade_date,
                "day_of_week": trade_date.strftime("%A"),
                "direction": "Short",
                "ticker": ticker,
                "shares": shares,
                "entry_time": str(entry_time_val)[:5],
                "exit_time": str(exit_time_val)[:5] if exit_time_val else None,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "gap_pct": round(gap_pct, 3),
                "avg_gap_14d": round(cand["avg_gap_14d"], 3),
                "prev_close": round(cand["prev_close"], 2),
                "initial_sl": round(initial_sl, 2),
                "sl_level": round(active_sl, 2),
                "lowest_price": round(lowest_since_entry, 2),
                "exit_reason": exit_reason,
                "pnl_pts": round(pnl_per_share, 2),
                "pnl_inr": round(net_pnl, 2),
                "portfolio_value": round(portfolio_value, 0),
                "trail_log": trail_log,
            })

    return pd.DataFrame(all_trades)


# ── Stats ────────────────────────────────────────────────────────────────────


def compute_portfolio_stats(trades_df: pd.DataFrame, cfg: dict) -> dict:
    """Compute stats for a multi-stock portfolio strategy."""
    if trades_df.empty:
        return {"total_trades": 0, "message": "No trades generated"}

    pnl = trades_df["pnl_inr"]
    wins = trades_df[pnl > 0]
    losses = trades_df[pnl <= 0]

    total = len(trades_df)
    win_rate = len(wins) / total * 100 if total > 0 else 0

    avg_win = wins["pnl_inr"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_inr"].mean() if len(losses) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    gross_wins = wins["pnl_inr"].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses["pnl_inr"].sum()) if len(losses) > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else (99.0 if gross_wins > 0 else 0)

    # Drawdown from daily aggregate P&L
    daily_pnl = trades_df.groupby("date")["pnl_inr"].sum().sort_index()
    capital = float(cfg["STARTING_CAPITAL"])
    equity = capital + daily_pnl.cumsum()
    running_max = equity.cummax()
    drawdown = running_max - equity
    max_dd = drawdown.max()
    max_dd_pct = (max_dd / running_max[drawdown.idxmax()] * 100) if max_dd > 0 else 0

    # Streaks
    flags = (pnl > 0).tolist()
    mws = mls = cw = cl = 0
    for v in flags:
        if v:
            cw += 1; cl = 0
        else:
            cl += 1; cw = 0
        mws = max(mws, cw)
        mls = max(mls, cl)

    # Per-share pnl stats (for compatibility with frontend)
    pnl_pts = trades_df["pnl_pts"]

    # Unique trading days
    n_days = trades_df["date"].nunique()
    avg_positions = total / n_days if n_days > 0 else 0

    # Return on capital
    total_pnl = pnl.sum()
    return_pct = total_pnl / capital * 100

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 1),
        "total_pnl_pts": round(pnl_pts.sum(), 2),
        "total_pnl_inr": round(total_pnl, 0),
        "avg_pnl_pts": round(pnl_pts.mean(), 2),
        "avg_win_pts": round(wins["pnl_pts"].mean(), 2) if len(wins) > 0 else 0,
        "avg_loss_pts": round(losses["pnl_pts"].mean(), 2) if len(losses) > 0 else 0,
        "rr_ratio": round(rr, 2),
        "profit_factor": round(pf, 2),
        "max_dd_inr": round(max_dd, 0),
        "max_dd_pct": round(max_dd_pct, 1),
        "max_dd_pts": round(max_dd / 1, 2),  # dd in INR as "pts" for compat
        "max_win_streak": mws,
        "max_loss_streak": mls,
        "long_trades": 0,
        "short_trades": total,
        "long_wr": 0,
        "short_wr": round(win_rate, 1),
        "exit_counts": trades_df["exit_reason"].value_counts().to_dict(),
        "best_trade_pts": round(pnl_pts.max(), 2),
        "worst_trade_pts": round(pnl_pts.min(), 2),
        "median_pnl_pts": round(pnl_pts.median(), 2),
        "std_pnl_pts": round(pnl_pts.std(ddof=0), 2) if len(trades_df) > 1 else 0,
        "avg_hold_mins": 0,
        # Portfolio-specific
        "starting_capital": capital,
        "final_portfolio_value": round(capital + total_pnl, 0),
        "return_on_capital_pct": round(return_pct, 2),
        "trading_days": n_days,
        "avg_positions_per_day": round(avg_positions, 1),
        "avg_gap_of_entries": round(trades_df["gap_pct"].mean(), 3),
        "unique_tickers_traded": trades_df["ticker"].nunique(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def run_strategy(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Execute the full strategy. Returns (trades_df, stats)."""
    print("Gap Short Reversal Strategy")
    print(f"  Date range: {cfg['DATE_FROM']} to {cfg['DATE_TO']}")
    print(f"  Capital: {cfg['STARTING_CAPITAL']:,.0f} | Position: {cfg['POSITION_SIZE_PCT']}%")
    print(f"  Min gap: {cfg['MIN_GAP_PCT']}% | Lookback: {cfg['GAP_LOOKBACK_DAYS']}d")
    print(f"  Max positions: {cfg['MAX_POSITIONS']}")
    print()

    # Load universe
    tickers = load_universe(cfg)
    print(f"Phase 1: Loading daily summaries for {len(tickers)} tickers...")

    # Extend date_from backwards for warmup
    lookback = int(cfg["GAP_LOOKBACK_DAYS"])
    warmup_from = parse_date(cfg["DATE_FROM"])
    # Go back ~45 calendar days to ensure enough trading days for warmup
    from datetime import timedelta
    warmup_start = (warmup_from - timedelta(days=lookback * 3)).strftime("%Y-%m-%d")

    summaries = build_daily_summaries(tickers, warmup_start, cfg["DATE_TO"])
    print(f"  Built summaries for {len(summaries)} tickers")

    print(f"\nPhase 2: Computing gap signals (lookback={lookback})...")
    enriched = compute_gap_signals(summaries, lookback)
    print(f"  {len(enriched)} tickers have enough data for gap analysis")

    print(f"\nPhase 3: Running day-by-day simulation...")
    trades_df = run_simulation(enriched, cfg)

    if trades_df.empty:
        stats = {
            "message": "No trades generated for the selected configuration.",
            "total_trades": 0,
        }
        print("\nNo trades generated.")
    else:
        stats = compute_portfolio_stats(trades_df, cfg)
        stats["strategy"] = "Gap Short Reversal"
        stats["notes"] = (
            f"Short stocks gapping > {cfg['MIN_GAP_PCT']}% above 14d avg gap. "
            f"Trailing SL = gap%. EOD exit {cfg['EOD_EXIT_TIME']}. "
            f"Max {cfg['MAX_POSITIONS']} positions."
        )
        print(
            f"\nDone: {stats['total_trades']} trades | "
            f"PnL {stats['total_pnl_inr']:,.0f} INR | "
            f"Win rate {stats['win_rate']}% | "
            f"Return {stats['return_on_capital_pct']}%"
        )

    return trades_df, stats


def main() -> None:
    cfg = get_config(DEFAULT_CONFIG)
    trades_df, stats = run_strategy(cfg)
    save_results(trades_df, stats, cfg)


if __name__ == "__main__":
    main()
