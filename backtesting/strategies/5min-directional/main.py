from __future__ import annotations

from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BACKTESTING_DIR = Path(__file__).resolve().parents[2]
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

import engine as shared_engine
from strategy_runtime import DATA_DIR, save_results, get_config


DEFAULT_CONFIG = {
    "SPOT_DATA_PATH": str(DATA_DIR / "spot_parquet"),
    "OPT_DATA_PATH": str(DATA_DIR / "options_parquet"),
    "DATE_FROM": "2025-07-01",
    "DATE_TO": "2025-12-31",
    "STRIKE_INTERVAL": 50,
    "LOT_SIZE": 75,
    "NUM_LOTS": 1,
    "EXPIRY_CODE": 1,
    "EXPIRY_CODE_ON_EXPIRY_DAY": 1,  # e.g. 2 → use code 2 on expiry days
    "SPOT_BAR_MINS": 5,
    # ── Signal candle ──
    "SIGNAL_CANDLE_TIME": "09:15",      # first candle timestamp
    # ── Body threshold ──
    "BODY_THRESHOLD": 20,               # min candle body (pts) to trigger
    # ── Direction filter ──
    "DIRECTION": "both",                # both | bull_only | bear_only
    # ── Strike selection ──
    "STRIKE_OFFSET": 0,                 # 0=ATM, 1=OTM, -1=ITM
    # ── Decay target ──
    "DECAY_TARGET_FRACTION": 0.25,      # exit when option = X% of entry (0=disabled)
    # ── Stop loss ──
    "OPTION_SL_MULT": 1.2,             # SL if option = X× entry (null=disabled)
    "USE_SPOT_SL": False,               # SL if spot violates candle extreme
    # ── EOD exit ──
    "EOD_EXIT_TIME": "15:00",
    # ── Costs ──
    "SLIPPAGE_POINTS": 0.5,
    "BROKERAGE_PER_TRADE": 40,
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def parse_date(text: str):
    return datetime.strptime(text, "%Y-%m-%d").date()


def parse_time(text: str):
    return datetime.strptime(text, "%H:%M").time()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = (DATA_DIR / path_text).resolve()
    if candidate.exists():
        return candidate
    return (DATA_DIR.parent / path_text).resolve()


def add_minutes(t: time, mins: int) -> time:
    dt = datetime(2000, 1, 1, t.hour, t.minute, t.second)
    return (dt + timedelta(minutes=mins)).time()


def get_strike(price, offset, interval):
    """ATM + offset strikes. offset=0→ATM, 1→OTM, -1→ITM."""
    atm = int(round(price / interval) * interval)
    return atm + offset * interval


def detect_expiry_dates(trading_dates):
    """Detect weekly expiry dates from the trading calendar.

    Nifty weekly expiry falls on Thursday. If Thursday is a holiday the
    last trading day before it (within the same week) becomes expiry.
    """
    weeks: dict[tuple, list] = defaultdict(list)
    for d in trading_dates:
        weeks[d.isocalendar()[:2]].append(d)

    expiry_dates: set = set()
    for _, days in weeks.items():
        days.sort()
        thursdays = [d for d in days if d.weekday() == 3]
        if thursdays:
            expiry_dates.add(thursdays[0])
        else:
            before_thu = [d for d in days if d.weekday() < 3]
            if before_thu:
                expiry_dates.add(max(before_thu))
    return expiry_dates


# ── Main backtest logic ─────────────────────────────────────────────────────


def build_trade_rows(cfg: dict) -> pd.DataFrame:
    start_date = parse_date(cfg["DATE_FROM"])
    end_date = parse_date(cfg["DATE_TO"])

    spot_path = str(resolve_path(cfg["SPOT_DATA_PATH"]))
    opt_path = str(resolve_path(cfg["OPT_DATA_PATH"]))

    spot_raw = shared_engine.load_dataset_folder(
        spot_path, cfg["DATE_FROM"], cfg["DATE_TO"], "Spot",
    )
    opts = shared_engine.load_dataset_folder(
        opt_path, cfg["DATE_FROM"], cfg["DATE_TO"], "Options",
    )

    # Expiry code config (per-day switching on expiry days)
    default_expiry_code = int(cfg.get("EXPIRY_CODE", 1))
    expiry_code_on_expiry = cfg.get("EXPIRY_CODE_ON_EXPIRY_DAY")
    if expiry_code_on_expiry is not None:
        expiry_code_on_expiry = int(expiry_code_on_expiry)

    # ── Config values ──
    signal_candle_t = parse_time(cfg["SIGNAL_CANDLE_TIME"])
    spot_bar_mins = int(cfg["SPOT_BAR_MINS"])
    entry_from_t = add_minutes(signal_candle_t, spot_bar_mins)
    eod_t = parse_time(cfg["EOD_EXIT_TIME"])
    direction_filter = str(cfg.get("DIRECTION", "both"))
    strike_offset = int(cfg.get("STRIKE_OFFSET", 0))
    interval = int(cfg.get("STRIKE_INTERVAL", 50))
    lot_size = int(cfg["LOT_SIZE"])
    num_lots = int(cfg["NUM_LOTS"])
    slippage = float(cfg["SLIPPAGE_POINTS"])
    body_threshold = float(cfg.get("BODY_THRESHOLD", 20))
    decay_target = float(cfg.get("DECAY_TARGET_FRACTION", 0.25))
    opt_sl_mult = cfg.get("OPTION_SL_MULT")
    if opt_sl_mult is not None:
        opt_sl_mult = float(opt_sl_mult)
    use_spot_sl = bool(cfg.get("USE_SPOT_SL", False))
    brk_pts = shared_engine.brokerage_in_pts(
        lot_size, num_lots, float(cfg["BROKERAGE_PER_TRADE"])
    )

    # Pre-split by date for performance
    spot_by_date = {
        d: grp.sort_values("time").reset_index(drop=True)
        for d, grp in spot_raw.groupby("date")
    }
    opts_by_date = {
        d: grp.reset_index(drop=True)
        for d, grp in opts.groupby("date")
    }

    # Detect expiry dates for DTE-based expiry code switching
    available_dates = sorted(spot_by_date.keys())
    expiry_dates = detect_expiry_dates(available_dates) if expiry_code_on_expiry is not None else set()

    trades: list[dict] = []

    for trade_date in available_dates:
        if trade_date < start_date or trade_date > end_date:
            continue

        # Per-day expiry code: use alternate code on expiry days
        if expiry_code_on_expiry is not None and trade_date in expiry_dates:
            expiry_code = expiry_code_on_expiry
        else:
            expiry_code = default_expiry_code

        day_spot = spot_by_date[trade_date]
        all_day_opts = opts_by_date.get(trade_date, pd.DataFrame())
        if all_day_opts.empty:
            continue
        day_opts = all_day_opts[all_day_opts["expiry_code"] == expiry_code].reset_index(drop=True)
        if day_opts.empty:
            continue

        # ── STEP 1: Read the first candle ──
        candle_rows = day_spot[day_spot["time"] == signal_candle_t]
        if candle_rows.empty:
            continue
        candle = candle_rows.iloc[0]

        body = abs(candle["close"] - candle["open"])
        is_bull = candle["close"] > candle["open"]
        direction_today = "Bull" if is_bull else "Bear"

        # ── STEP 2: Body threshold filter ──
        if body < body_threshold:
            continue

        # ── Direction filter ──
        if direction_filter == "bull_only" and not is_bull:
            continue
        if direction_filter == "bear_only" and is_bull:
            continue

        # ── STEP 3: Determine option to sell ──
        opt_type = "PE" if is_bull else "CE"

        # Strike with direction-aware offset
        if opt_type == "PE":
            strike = get_strike(candle["close"], -strike_offset, interval)
        else:
            strike = get_strike(candle["close"], strike_offset, interval)

        # Spot SL levels
        spot_sl_bull = float(candle["low"])
        spot_sl_bear = float(candle["high"])

        # ── STEP 4: Find entry bar ──
        day_opt_f = day_opts[
            (day_opts["actual_strike"] == strike)
            & (day_opts["option_type"] == opt_type)
        ].sort_values("time").reset_index(drop=True)

        entry_candidates = day_opt_f[day_opt_f["time"] >= entry_from_t]
        if entry_candidates.empty:
            continue

        raw_entry = float(entry_candidates.iloc[0]["close"])
        entry_time = entry_candidates.iloc[0]["time"]
        if raw_entry <= 0:
            continue

        # For selling, we receive less than quoted close
        entry_price = raw_entry - slippage

        # ── STEP 5: Set exit levels ──
        target_price = (entry_price * decay_target) if decay_target > 0 else None
        opt_sl_price = (entry_price * opt_sl_mult) if opt_sl_mult else None

        # ── STEP 6: Walk forward through remaining 1-min option bars ──
        post_opt = day_opt_f[
            (day_opt_f["time"] > entry_time)
            & (day_opt_f["time"] <= eod_t)
        ]
        post_spot = day_spot[
            (day_spot["time"] > signal_candle_t)
            & (day_spot["time"] <= eod_t)
        ]

        exit_price = None
        exit_time = None
        exit_reason = "EOD"

        for _, obar in post_opt.iterrows():
            curr_t = obar["time"]
            opt_high = obar["high"]
            opt_low = obar["low"]

            # A: Spot SL check (lookahead-free)
            spot_sl_hit = False
            if use_spot_sl:
                completed_cutoff = add_minutes(curr_t, -spot_bar_mins)
                completed_spot = post_spot[post_spot["time"] <= completed_cutoff]
                if not completed_spot.empty:
                    if is_bull and completed_spot["low"].min() <= spot_sl_bull:
                        spot_sl_hit = True
                    if not is_bull and completed_spot["high"].max() >= spot_sl_bear:
                        spot_sl_hit = True

            # B: Option SL check
            opt_sl_hit = (opt_sl_price is not None) and (opt_high >= opt_sl_price)

            # C: Target check
            tgt_hit = (target_price is not None) and (opt_low <= target_price)

            # D: Conflict resolution — SL wins over target in same bar
            any_sl = opt_sl_hit or spot_sl_hit

            if tgt_hit and any_sl:
                exit_price = float(obar["open"])
                exit_time = curr_t
                exit_reason = "SL"
                break
            elif tgt_hit:
                exit_price = target_price
                exit_time = curr_t
                exit_reason = "TARGET"
                break
            elif any_sl:
                exit_price = float(obar["open"])
                exit_time = curr_t
                exit_reason = "SL_OPT" if opt_sl_hit else "SL_SPOT"
                break
            else:
                exit_price = float(obar["close"])
                exit_time = curr_t

        if exit_price is None:
            exit_price = entry_price
            exit_time = entry_time
            exit_reason = "NO_DATA"

        # ── STEP 7: P&L ──
        exit_price_slip = exit_price + slippage
        raw_pnl = entry_price - exit_price_slip
        pnl_pts = raw_pnl - brk_pts
        pnl_inr = pnl_pts * lot_size * num_lots

        trades.append({
            "date": trade_date,
            "day_of_week": trade_date.strftime("%A"),
            "direction": direction_today,
            "position_side": "short_option",
            "trade_mode": "sell",
            "opt_type": opt_type,
            "strike": strike,
            "atm": get_strike(candle["close"], 0, interval),
            "signal_time": signal_candle_t,
            "trigger_time": signal_candle_t,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "or_high": round(float(candle["high"]), 2),
            "or_low": round(float(candle["low"]), 2),
            "spot_sl": round(spot_sl_bull if is_bull else spot_sl_bear, 2),
            "spot_at_signal": round(float(candle["close"]), 2),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "target_price": round(target_price, 2) if target_price else None,
            "sl_level": round(opt_sl_price, 2) if opt_sl_price else None,
            "exit_reason": exit_reason,
            "pnl_pts": round(pnl_pts, 2),
            "pnl_inr": round(pnl_inr, 2),
            "body_pts": round(body, 1),
            "candle_open": round(float(candle["open"]), 2),
            "candle_close": round(float(candle["close"]), 2),
            "candle_high": round(float(candle["high"]), 2),
            "candle_low": round(float(candle["low"]), 2),
            "filled_exit_price": round(exit_price_slip, 2),
            "expiry_code": expiry_code,
        })

    return pd.DataFrame(trades)


def main() -> None:
    cfg = get_config(DEFAULT_CONFIG)
    trades_df = build_trade_rows(cfg)

    if trades_df.empty:
        stats = {
            "message": "No 5-min directional trades were generated for the selected range.",
            "total_trades": 0,
        }
        print(stats["message"])
        save_results(trades_df, stats, cfg)
        return

    stats = shared_engine.compute_stats(trades_df, cfg)
    stats["strategy"] = "First 5-Min Candle Directional (Sell)"
    stats["notes"] = (
        f"Sell opposite option on first 5-min candle direction. "
        f"Body >= {cfg.get('BODY_THRESHOLD', 20)}pts. "
        f"Decay target: {cfg.get('DECAY_TARGET_FRACTION', 0.25)}. "
        f"Direction: {cfg.get('DIRECTION', 'both')}."
    )

    print(
        f"5-Min Directional backtest complete: {len(trades_df)} trades | "
        f"PnL {stats.get('total_pnl_pts', 0)} pts"
    )
    save_results(trades_df, stats, cfg)


if __name__ == "__main__":
    main()
