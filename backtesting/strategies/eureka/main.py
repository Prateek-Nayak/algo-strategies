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
    "DATE_FROM": "2025-12-01",
    "DATE_TO": "2025-12-31",
    "STRIKE_INTERVAL": 50,
    "LOT_SIZE": 65,
    "NUM_LOTS": 1,
    "EXPIRY_CODE": 1,
    "EXPIRY_CODE_ON_EXPIRY_DAY": 1,
    # ── Pair & signal ──
    "SPOT_BAR_MINS": 5,
    "MAX_RANGE": 100,            # skip pairs > 100 pts range
    "MIN_PREMIUM": 3,            # minimum option premium to enter
    # ── Stop loss ──
    "OPT_SL_POINTS": 30,        # option premium stop loss (pts)
    "SL_EXIT_PRICE_MODE": "close",  # close | open | high (bar field on SL exit)
    # ── Exit ──
    "LAST_SIGNAL_TIME": "15:00",
    "SQUARE_OFF_TIME": "15:20",
    "CONSEC_SL_LIMIT": 3,       # stop trading after N consecutive SLs
    # ── Trail ──
    "TRAIL_ENABLED": True,       # move SL to breakeven after 1R profit
    # ── Costs ──
    "SLIPPAGE_POINTS": 0.5,
    "BROKERAGE_PER_TRADE": 40,
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def parse_date(text: str):
    return datetime.strptime(text, "%Y-%m-%d").date()


def parse_time_str(text: str):
    return datetime.strptime(text, "%H:%M").time()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = (DATA_DIR / path_text).resolve()
    if candidate.exists():
        return candidate
    return (DATA_DIR.parent / path_text).resolve()


def detect_expiry_dates(trading_dates):
    """Detect weekly expiry dates from the trading calendar."""
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


# ── Option index (fast lookup) ──────────────────────────────────────────────


def build_option_index(opts_df: pd.DataFrame) -> dict:
    """Build a (date, time, strike, option_type) -> {open, high, low, close} dict."""
    dates = opts_df["date"].values
    times = opts_df["time"].values
    strikes = opts_df["actual_strike"].values.astype(int)
    otypes = opts_df["option_type"].values
    opens = opts_df["open"].values.astype(float)
    highs = opts_df["high"].values.astype(float)
    lows = opts_df["low"].values.astype(float)
    closes = opts_df["close"].values.astype(float)

    idx: dict = {}
    for i in range(len(opts_df)):
        idx[(dates[i], times[i], strikes[i], otypes[i])] = (
            opens[i], highs[i], lows[i], closes[i],
        )
    return idx


def lookup_opt_atm(idx: dict, date, t, spot: float, otype: str, interval: int):
    """Lookup ATM option, falling back to +/-1 strike."""
    atm = int(round(spot / interval) * interval)
    for s in [atm, atm + interval, atm - interval]:
        k = (date, t, s, otype)
        if k in idx:
            o, h, l, c = idx[k]
            return {"open": o, "high": h, "low": l, "close": c, "strike": s}
    return None


def lookup_opt_strike(idx: dict, date, t, strike: int, otype: str):
    """Lookup a specific strike."""
    k = (date, t, strike, otype)
    if k in idx:
        o, h, l, c = idx[k]
        return {"open": o, "high": h, "low": l, "close": c, "strike": strike}
    return None


# ── Trade row builder ───────────────────────────────────────────────────────


def _make_trade_row(
    trade: dict,
    trade_date,
    exit_time,
    exit_price: float,
    filled_exit_price: float,
    pnl_pts: float,
    pnl_inr: float,
    exit_reason: str,
    spot_at_exit: float,
    expiry_code: int,
    opt_sl_pts: float,
) -> dict:
    """Build a standardized trade row dict."""
    return {
        "date": trade_date,
        "day_of_week": trade_date.strftime("%A"),
        "direction": trade["dir"],
        "position_side": "short_option",
        "trade_mode": "sell",
        "opt_type": trade["option_type"],
        "strike": trade["strike"],
        "atm": trade.get("atm"),
        "signal_time": trade["signal_time"],
        "trigger_time": trade["signal_time"],
        "entry_time": trade["entry_time"],
        "exit_time": exit_time,
        "or_high": round(trade["pair_high"], 2),
        "or_low": round(trade["pair_low"], 2),
        "spot_sl": round(trade["sl_spot"], 2),
        "spot_at_signal": round(trade["entry_spot"], 2),
        "entry_price": trade["raw_entry"],
        "exit_price": round(exit_price, 2),
        "target_price": None,
        "sl_level": round(trade["raw_entry"] + opt_sl_pts, 2),
        "exit_reason": exit_reason,
        "pnl_pts": round(pnl_pts, 2),
        "pnl_inr": round(pnl_inr, 2),
        "pair_type": trade["pair_type"],
        "pair_range": trade["pair_range"],
        "trailing_active": trade.get("trailing_active", False),
        "filled_exit_price": round(filled_exit_price, 2),
        "expiry_code": expiry_code,
    }


# ── Main backtest logic ─────────────────────────────────────────────────────


def build_trade_rows(cfg: dict) -> pd.DataFrame:
    start_date = parse_date(cfg["DATE_FROM"])
    end_date = parse_date(cfg["DATE_TO"])

    spot_path = str(resolve_path(cfg["SPOT_DATA_PATH"]))
    opt_path = str(resolve_path(cfg["OPT_DATA_PATH"]))

    spot_raw = shared_engine.load_dataset_folder(
        spot_path, cfg["DATE_FROM"], cfg["DATE_TO"], "Spot",
    )
    opts_raw = shared_engine.load_dataset_folder(
        opt_path, cfg["DATE_FROM"], cfg["DATE_TO"], "Options",
    )
    spot_5m = shared_engine.resample_spot(spot_raw, int(cfg["SPOT_BAR_MINS"]))

    # Config values
    default_expiry_code = int(cfg.get("EXPIRY_CODE", 1))
    expiry_code_on_expiry = cfg.get("EXPIRY_CODE_ON_EXPIRY_DAY")
    if expiry_code_on_expiry is not None:
        expiry_code_on_expiry = int(expiry_code_on_expiry)

    interval = int(cfg["STRIKE_INTERVAL"])
    lot_size = int(cfg["LOT_SIZE"])
    num_lots = int(cfg["NUM_LOTS"])
    slippage = float(cfg["SLIPPAGE_POINTS"])
    brk_pts = shared_engine.brokerage_in_pts(
        lot_size, num_lots, float(cfg["BROKERAGE_PER_TRADE"])
    )
    max_range = float(cfg["MAX_RANGE"])
    min_prem = float(cfg["MIN_PREMIUM"])
    opt_sl_pts = float(cfg["OPT_SL_POINTS"])
    sl_exit_mode = str(cfg.get("SL_EXIT_PRICE_MODE", "close")).lower()
    last_signal_t = parse_time_str(cfg["LAST_SIGNAL_TIME"])
    square_off_t = parse_time_str(cfg["SQUARE_OFF_TIME"])
    consec_sl_limit = int(cfg["CONSEC_SL_LIMIT"])
    trail_enabled = bool(cfg.get("TRAIL_ENABLED", True))

    # Detect expiry dates
    available_dates = sorted(spot_5m["date"].unique())
    expiry_dates = (
        detect_expiry_dates(available_dates)
        if expiry_code_on_expiry is not None
        else set()
    )

    # Pre-split spot by date
    spot_by_date = {
        d: grp.sort_values("time").reset_index(drop=True)
        for d, grp in spot_5m.groupby("date")
    }

    # Build option index (all expiry codes; we filter at lookup time)
    print("  Building option index...")
    opt_idx = build_option_index(opts_raw)
    print(f"  Option index: {len(opt_idx):,} entries")

    trades: list[dict] = []

    for trade_date in available_dates:
        if trade_date < start_date or trade_date > end_date:
            continue

        day_df = spot_by_date.get(trade_date)
        if day_df is None or len(day_df) < 3:
            continue

        # Per-day expiry code
        if expiry_code_on_expiry is not None and trade_date in expiry_dates:
            expiry_code = expiry_code_on_expiry
        else:
            expiry_code = default_expiry_code

        consec_sl = 0
        active_trade = None
        pair_scan_from = 0
        current_pair = None

        times_arr = day_df["time"].values
        opens_arr = day_df["open"].values.astype(float)
        highs_arr = day_df["high"].values.astype(float)
        lows_arr = day_df["low"].values.astype(float)
        closes_arr = day_df["close"].values.astype(float)

        for ci in range(len(day_df)):
            ct = times_arr[ci]
            spot_o = opens_arr[ci]
            spot_h = highs_arr[ci]
            spot_l = lows_arr[ci]
            spot_c = closes_arr[ci]

            # ── 3-SL RULE ──
            if consec_sl >= consec_sl_limit:
                break

            # ── EXIT logic (if in a trade) ──
            if active_trade is not None:

                # Square off at EOD
                if ct >= square_off_t:
                    co = lookup_opt_strike(
                        opt_idx, trade_date, ct,
                        active_trade["strike"], active_trade["option_type"],
                    )
                    ep = co["close"] if co else active_trade["raw_entry"]
                    ep_slip = ep + slippage
                    raw_pnl = active_trade["raw_entry"] - ep
                    pnl_pts = (active_trade["raw_entry"] - ep_slip) - brk_pts
                    pnl_inr = pnl_pts * lot_size * num_lots

                    trades.append(_make_trade_row(
                        active_trade, trade_date, ct, ep, ep_slip,
                        pnl_pts, pnl_inr, "SQUARE_OFF", spot_c, expiry_code,
                        opt_sl_pts,
                    ))
                    if raw_pnl < 0:
                        consec_sl += 1
                    else:
                        consec_sl = 0
                    pair_scan_from = ci + 1
                    current_pair = None
                    active_trade = None
                    continue

                co = lookup_opt_strike(
                    opt_idx, trade_date, ct,
                    active_trade["strike"], active_trade["option_type"],
                )
                if co:
                    # Spot-based SL check
                    sl_hit = False
                    if active_trade["dir"] == "Bull" and spot_l <= active_trade["sl_spot"]:
                        sl_hit = True
                    if active_trade["dir"] == "Bear" and spot_h >= active_trade["sl_spot"]:
                        sl_hit = True

                    if sl_hit:
                        ep = co.get(sl_exit_mode, co["close"])
                        ep_slip = ep + slippage
                        pnl_pts = (active_trade["raw_entry"] - ep_slip) - brk_pts
                        pnl_inr = pnl_pts * lot_size * num_lots

                        trades.append(_make_trade_row(
                            active_trade, trade_date, ct, ep, ep_slip,
                            pnl_pts, pnl_inr, "SL_SPOT", spot_c, expiry_code,
                            opt_sl_pts,
                        ))
                        consec_sl += 1
                        pair_scan_from = ci + 1
                        current_pair = None
                        active_trade = None
                        continue

                    # Option premium SL
                    if co["high"] >= active_trade["raw_entry"] + opt_sl_pts:
                        ep = active_trade["raw_entry"] + opt_sl_pts
                        ep_slip = ep + slippage
                        pnl_pts = (active_trade["raw_entry"] - ep_slip) - brk_pts
                        pnl_inr = pnl_pts * lot_size * num_lots

                        trades.append(_make_trade_row(
                            active_trade, trade_date, ct, ep, ep_slip,
                            pnl_pts, pnl_inr, "SL_OPT", spot_c, expiry_code,
                            opt_sl_pts,
                        ))
                        consec_sl += 1
                        pair_scan_from = ci + 1
                        current_pair = None
                        active_trade = None
                        continue

                    # TRAIL: move SL to breakeven after 1R profit
                    if trail_enabled:
                        risk = active_trade["pair_range"]
                        if (active_trade["dir"] == "Bull"
                                and spot_h >= active_trade["entry_spot"] + risk):
                            active_trade["sl_spot"] = active_trade["entry_spot"]
                            active_trade["trailing_active"] = True
                        if (active_trade["dir"] == "Bear"
                                and spot_l <= active_trade["entry_spot"] - risk):
                            active_trade["sl_spot"] = active_trade["entry_spot"]
                            active_trade["trailing_active"] = True

            # ── FIND PAIR & ENTRY (if no trade) ──
            if active_trade is None and ct < last_signal_t:

                # Step 1: Scan for RG/GR pair
                if current_pair is None and ci >= pair_scan_from:
                    if ci >= pair_scan_from + 1:
                        prev_o = opens_arr[ci - 1]
                        prev_c = closes_arr[ci - 1]
                        prev_h = highs_arr[ci - 1]
                        prev_l = lows_arr[ci - 1]

                        is_rg = prev_c < prev_o and spot_c > spot_o
                        is_gr = prev_c > prev_o and spot_c < spot_o

                        if is_rg or is_gr:
                            ph = max(prev_h, spot_h)
                            pl = min(prev_l, spot_l)
                            pr = ph - pl
                            if pr <= max_range:
                                current_pair = {
                                    "idx1": ci - 1,
                                    "idx2": ci,
                                    "pair_high": ph,
                                    "pair_low": pl,
                                    "pair_range": round(pr, 2),
                                    "type": "RG" if is_rg else "GR",
                                }

                # Step 2: Check breakout (candle CLOSES above/below range)
                elif current_pair is not None and ci > current_pair["idx2"]:
                    p = current_pair

                    breakout_dir = None
                    if spot_c >= p["pair_high"]:
                        breakout_dir = "Bull"
                    elif spot_c <= p["pair_low"]:
                        breakout_dir = "Bear"

                    if breakout_dir:
                        # Entry at NEXT candle's option OPEN
                        if ci + 1 >= len(day_df):
                            current_pair = None
                            continue

                        next_ct = times_arr[ci + 1]
                        next_spot_c = closes_arr[ci + 1]

                        otype = "PE" if breakout_dir == "Bull" else "CE"

                        # Look up ATM option for the right expiry code
                        od = _lookup_opt_with_expiry(
                            opt_idx, opts_raw, trade_date, next_ct,
                            next_spot_c, otype, interval, expiry_code,
                        )

                        if od and od["open"] > min_prem:
                            raw_entry = od["open"]
                            entry_price = raw_entry - slippage
                            next_spot_o = opens_arr[ci + 1]

                            active_trade = {
                                "dir": breakout_dir,
                                "option_type": otype,
                                "strike": od["strike"],
                                "raw_entry": round(entry_price, 2),
                                "entry_time": next_ct,
                                "entry_spot": next_spot_o,
                                "sl_spot": (
                                    p["pair_low"] if breakout_dir == "Bull"
                                    else p["pair_high"]
                                ),
                                "pair_high": p["pair_high"],
                                "pair_low": p["pair_low"],
                                "pair_range": p["pair_range"],
                                "pair_type": p["type"],
                                "trailing_active": False,
                                "signal_time": ct,
                                "atm": int(round(next_spot_c / interval) * interval),
                            }
                            current_pair = None
                            continue
                        else:
                            current_pair = None

        # EOD close any open trade
        if active_trade is not None:
            last_ct = times_arr[-1]
            last_spot_c = closes_arr[-1]
            co = lookup_opt_strike(
                opt_idx, trade_date, last_ct,
                active_trade["strike"], active_trade["option_type"],
            )
            ep = co["close"] if co else active_trade["raw_entry"]
            ep_slip = ep + slippage
            pnl_pts = (active_trade["raw_entry"] - ep_slip) - brk_pts
            pnl_inr = pnl_pts * lot_size * num_lots

            trades.append(_make_trade_row(
                active_trade, trade_date, last_ct, ep, ep_slip,
                pnl_pts, pnl_inr, "SQUARE_OFF", last_spot_c, expiry_code,
                opt_sl_pts,
            ))

    return pd.DataFrame(trades)


def _lookup_opt_with_expiry(
    opt_idx, opts_raw, trade_date, t, spot, otype, interval, expiry_code,
):
    """ATM lookup that verifies the strike belongs to the desired expiry code."""
    atm = int(round(spot / interval) * interval)
    for s in [atm, atm + interval, atm - interval]:
        k = (trade_date, t, s, otype)
        if k not in opt_idx:
            continue
        # Verify expiry code via raw data
        match = opts_raw[
            (opts_raw["date"] == trade_date)
            & (opts_raw["time"] == t)
            & (opts_raw["actual_strike"] == s)
            & (opts_raw["option_type"] == otype)
            & (opts_raw["expiry_code"] == expiry_code)
        ]
        if match.empty:
            continue
        o, h, l, c = opt_idx[k]
        return {"open": o, "high": h, "low": l, "close": c, "strike": s}
    return None


def main() -> None:
    cfg = get_config(DEFAULT_CONFIG)
    trades_df = build_trade_rows(cfg)

    if trades_df.empty:
        stats = {
            "message": "No Traffic Light trades were generated for the selected range.",
            "total_trades": 0,
        }
        print(stats["message"])
        save_results(trades_df, stats, cfg)
        return

    stats = shared_engine.compute_stats(trades_df, cfg)
    stats["strategy"] = "Traffic Light (Eureka)"
    stats["notes"] = (
        f"Red-Green / Green-Red pair breakout on {cfg['SPOT_BAR_MINS']}-min candles. "
        f"Max pair range: {cfg['MAX_RANGE']}pts. "
        f"Option SL: {cfg['OPT_SL_POINTS']}pts. "
        f"Trail: {'on' if cfg.get('TRAIL_ENABLED', True) else 'off'}. "
        f"SL exit mode: {cfg.get('SL_EXIT_PRICE_MODE', 'close')}."
    )

    print(
        f"Traffic Light backtest complete: {len(trades_df)} trades | "
        f"PnL {stats.get('total_pnl_pts', 0)} pts"
    )
    save_results(trades_df, stats, cfg)


if __name__ == "__main__":
    main()
