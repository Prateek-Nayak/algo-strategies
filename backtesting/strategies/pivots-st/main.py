from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
import sys

import pandas as pd
import numpy as np

BACKTESTING_DIR = Path(__file__).resolve().parents[2]
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

# Import supertrend calculator from the live strategy module
_ST_DIR = str(Path(__file__).resolve().parents[3] / "strategy" / "Pivot + ST")
if _ST_DIR not in sys.path:
    sys.path.insert(0, _ST_DIR)

import engine as shared_engine
from strategy_runtime import DATA_DIR, save_results, get_config
from supertrend import calculate_supertrend_tradingview


DEFAULT_CONFIG = {
    "SPOT_DATA_PATH": str(DATA_DIR / "spot_parquet"),
    "OPT_DATA_PATH": str(DATA_DIR / "options_parquet"),
    "DATE_FROM": "2025-01-01",
    "DATE_TO": "2025-12-31",
    "SPOT_CANDLE_MINUTES": 5,
    "ENTRY_START_TIME": "10:00",
    "LAST_SIGNAL_TIME": "14:59",
    "EOD_EXIT_TIME": "15:00",
    "EXPIRY_CODE": 1,
    "LOT_SIZE": 75,
    "NUM_LOTS": 1,
    "TARGET_POINTS": 50,
    "STOP_LOSS_POINTS": 50,
    "BROKERAGE_PER_TRADE": 40,
    "SLIPPAGE_POINTS": 0.5,
    "ENTRY_MODE": "st",
    "ST_PERIOD": 10,
    "ST_MULTIPLIER": 3.0,
    "ST_EXIT_MODE": "spot_and_option",
    "ST_REENTRY_SL": 20,
    "ST_REENTRY_TGT": "EOD",
}


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


def compute_pivots(high: float, low: float, close: float) -> dict[str, float]:
    pivot = (high + low + close) / 3
    levels = {"pivot": round(pivot, 2)}
    for index in range(1, 11):
        resistance = high + index * (pivot - low)
        support = low - index * (high - pivot)
        if index == 1:
            resistance = 2 * pivot - low
            support = 2 * pivot - high
        elif index == 2:
            resistance = pivot + (high - low)
            support = pivot - (high - low)
        levels[f"r{index}"] = round(resistance, 2)
        levels[f"s{index}"] = round(support, 2)
    return levels


def should_trigger_trade(candle: dict[str, float], pivots: dict[str, float]):
    touched = []
    for name, level in pivots.items():
        if candle["low"] <= level <= candle["high"]:
            touched.append((name, level))
    return touched


def calculate_strike(spot_close: float, option_type: str) -> int:
    if option_type == "CE":
        return int((spot_close // 50) * 50)
    return int(((spot_close + 49.9999) // 50) * 50)


def shift_time_by_minutes(value: time, minutes: int) -> time:
    base = datetime.combine(datetime(2000, 1, 1).date(), value)
    return (base + timedelta(minutes=minutes)).time()


def detect_expiry_dates(trading_dates):
    """Detect weekly expiry dates from the trading calendar."""
    from collections import defaultdict

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


def choose_exit_reason_for_short(bar: dict, sl_price: float, tgt_price: float) -> tuple[str, float]:
    sl_hit = bar["high"] >= sl_price
    tgt_hit = bar["low"] <= tgt_price

    if sl_hit and tgt_hit:
        open_price = float(bar["open"])
        if abs(open_price - sl_price) <= abs(open_price - tgt_price):
            return "OPT_SL", sl_price
        return "TARGET", tgt_price
    if sl_hit:
        return "OPT_SL", sl_price
    if tgt_hit:
        return "TARGET", tgt_price
    return "", 0.0


# ── Supertrend helpers ──────────────────────────────────────────────────────


def resample_to_minutes(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample intraday OHLC data to the given minute interval."""
    if df.empty or "datetime_ist" not in df.columns:
        return df
    r = (
        df.set_index("datetime_ist")
        .resample(f"{minutes}min", closed="left", label="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open"])
        .reset_index()
    )
    r["date"] = r["datetime_ist"].dt.date
    r["time"] = r["datetime_ist"].dt.time
    return r.sort_values("datetime_ist").reset_index(drop=True)


def compute_st_directions(df: pd.DataFrame, period: int, multiplier: float) -> np.ndarray:
    """Return an array of 1 (bullish) / -1 (bearish) / NaN (not ready)."""
    n = len(df)
    if n < period:
        return np.full(n, np.nan)
    st_result = calculate_supertrend_tradingview(
        df[["open", "high", "low", "close"]].reset_index(drop=True).copy(),
        period=period,
        multiplier=multiplier,
    )
    directions = st_result["trend_direction"].values.astype(float)
    # Mark as NaN where supertrend value itself is NaN (warm-up period)
    directions[st_result["supertrend"].isna().values] = np.nan
    return directions


# ── Main backtest logic ─────────────────────────────────────────────────────


def build_trade_rows(cfg: dict) -> pd.DataFrame:
    start_date = parse_date(cfg["DATE_FROM"])
    end_date = parse_date(cfg["DATE_TO"])
    extended_start = (start_date - timedelta(days=10)).strftime("%Y-%m-%d")

    spot_path = str(resolve_path(cfg["SPOT_DATA_PATH"]))
    opt_path = str(resolve_path(cfg["OPT_DATA_PATH"]))

    spot_raw = shared_engine.load_dataset_folder(
        spot_path, extended_start, cfg["DATE_TO"], "Spot",
    )
    opts = shared_engine.load_dataset_folder(
        opt_path, extended_start, cfg["DATE_TO"], "Options",
    )
    candle_minutes = int(cfg["SPOT_CANDLE_MINUTES"])
    spot_5m = shared_engine.resample_spot(spot_raw, candle_minutes)

    # ── Entry mode ──
    entry_mode = str(cfg.get("ENTRY_MODE", "st")).strip().lower()
    if entry_mode not in ("st", "candle_color"):
        raise ValueError("ENTRY_MODE must be one of: st, candle_color")

    # ── Supertrend on spot (across all days for natural warm-up) ──
    st_period = int(cfg["ST_PERIOD"])
    st_multiplier = float(cfg["ST_MULTIPLIER"])
    st_exit_mode = str(cfg.get("ST_EXIT_MODE", "spot_and_option")).strip().lower()
    if st_exit_mode not in ("none", "spot_only", "spot_and_option", "reverse"):
        raise ValueError("ST_EXIT_MODE must be one of: none, spot_only, spot_and_option, reverse")
    # candle_color mode ignores ST for exits too
    if entry_mode == "candle_color":
        st_exit_mode = "none"

    # Re-entry config (only used when st_exit_mode == "reverse")
    _re_sl_raw = cfg.get("ST_REENTRY_SL", 20)
    _re_tgt_raw = cfg.get("ST_REENTRY_TGT", "EOD")
    reentry_sl_pts = None if str(_re_sl_raw).upper() == "EOD" else float(_re_sl_raw)
    reentry_tgt_pts = None if str(_re_tgt_raw).upper() == "EOD" else float(_re_tgt_raw)
    spot_5m = spot_5m.reset_index(drop=True)
    # Only compute ST when needed
    if entry_mode == "st":
        spot_5m["st_direction"] = compute_st_directions(spot_5m, st_period, st_multiplier)
    else:
        spot_5m["st_direction"] = np.nan

    entry_start = parse_time(cfg["ENTRY_START_TIME"])
    last_signal = parse_time(cfg["LAST_SIGNAL_TIME"])
    eod_exit = parse_time(cfg["EOD_EXIT_TIME"])
    slippage = float(cfg["SLIPPAGE_POINTS"])
    stop_loss_pts = float(cfg["STOP_LOSS_POINTS"])
    target_pts = float(cfg["TARGET_POINTS"])
    lot_size = int(cfg["LOT_SIZE"])
    num_lots = int(cfg["NUM_LOTS"])
    brokerage_pts = shared_engine.brokerage_in_pts(
        lot_size, num_lots, float(cfg["BROKERAGE_PER_TRADE"])
    )

    available_dates = sorted(spot_raw["date"].unique())
    default_expiry_code = cfg.get("EXPIRY_CODE", 1)
    expiry_dates = detect_expiry_dates(available_dates)
    trades: list[dict] = []

    for trade_date in available_dates:
        if trade_date < start_date or trade_date > end_date:
            continue

        previous_dates = [v for v in available_dates if v < trade_date]
        if not previous_dates:
            continue

        prev_date = previous_dates[-1]
        prev_day_spot = (
            spot_raw[spot_raw["date"] == prev_date]
            .sort_values("time")
            .reset_index(drop=True)
        )
        current_spot = (
            spot_5m[spot_5m["date"] == trade_date]
            .sort_values("time")
            .reset_index(drop=True)
        )
        expiry_code = 2 if trade_date in expiry_dates else default_expiry_code
        current_opts = (
            opts[(opts["date"] == trade_date) & (opts["expiry_code"] == expiry_code)]
            .sort_values("time")
            .reset_index(drop=True)
        )

        if prev_day_spot.empty or current_spot.empty or current_opts.empty:
            continue

        pivots = compute_pivots(
            float(prev_day_spot["high"].max()),
            float(prev_day_spot["low"].min()),
            float(prev_day_spot["close"].iloc[-1]),
        )

        # Spot ST lookup for today: time → direction (1 / -1)
        spot_st_map: dict[time, int] = {}
        for _, row in current_spot.iterrows():
            st_val = row.get("st_direction")
            if pd.notna(st_val):
                spot_st_map[row["time"]] = int(st_val)

        trade_taken = False
        for _, signal_bar in current_spot.iterrows():
            if trade_taken:
                break

            signal_time = signal_bar["time"]
            if signal_time < entry_start or signal_time >= last_signal:
                continue

            candle = {
                "open": float(signal_bar["open"]),
                "high": float(signal_bar["high"]),
                "low": float(signal_bar["low"]),
                "close": float(signal_bar["close"]),
            }
            touched_levels = should_trigger_trade(candle, pivots)
            if not touched_levels:
                continue

            # ── Direction ──
            if entry_mode == "st":
                # Supertrend: finalized after candle close; entry at next open.
                st_dir = spot_st_map.get(signal_time)
                if st_dir is None:
                    continue  # ST not ready yet, skip
                # ST bearish (-1) → sell CE (Short)
                # ST bullish (+1) → sell PE (Long)
                opt_type = "CE" if st_dir == -1 else "PE"
            else:
                # Candle colour: red candle → sell CE, green → sell PE
                st_dir = 0  # placeholder, not used for exits
                opt_type = "CE" if candle["close"] < candle["open"] else "PE"

            direction = "Long" if opt_type == "PE" else "Short"

            strike = calculate_strike(candle["close"], opt_type)
            entry_lookup_time = shift_time_by_minutes(
                signal_time, candle_minutes
            )

            # ── Get option data for this contract ──
            option_day = current_opts[
                (current_opts["actual_strike"] == strike)
                & (current_opts["option_type"] == opt_type)
            ].sort_values("time")
            if option_day.empty:
                continue

            entry_candidates = option_day[option_day["time"] >= entry_lookup_time]
            if entry_candidates.empty:
                continue

            entry_bar = entry_candidates.iloc[0]
            raw_entry = float(entry_bar["open"])
            entry_time = entry_bar["time"]
            entry_price = shared_engine.apply_slippage(
                raw_entry, "entry_sell", slippage
            )

            sl_price = entry_price + stop_loss_pts
            tgt_price = entry_price - target_pts

            # ── Resample option to 5-min ──
            # Use all available days for this contract (up to trade_date) as warm-up
            opt_warmup = opts[
                (opts["date"] <= trade_date)
                & (opts["expiry_code"] == expiry_code)
                & (opts["actual_strike"] == strike)
                & (opts["option_type"] == opt_type)
            ].sort_values("datetime_ist")
            opt_warmup_5m = resample_to_minutes(opt_warmup, candle_minutes)

            # Compute option ST only when needed for exit
            opt_st_map: dict[time, int] = {}
            if st_exit_mode == "spot_and_option":
                opt_warmup_st = compute_st_directions(opt_warmup_5m, st_period, st_multiplier)
                for oi in range(len(opt_warmup_5m)):
                    val = opt_warmup_st[oi]
                    if pd.notna(val) and opt_warmup_5m.iloc[oi]["date"] == trade_date:
                        opt_st_map[opt_warmup_5m.iloc[oi]["time"]] = int(val)

            # Extract only today's bars
            opt_5m = opt_warmup_5m[opt_warmup_5m["date"] == trade_date].reset_index(drop=True)

            # ── Exit loop on 5-min option bars ──
            future_opt = opt_5m[opt_5m["time"] > entry_time].to_dict("records")

            exit_reason = "EOD"
            exit_time = entry_time
            exit_raw = raw_entry
            st_exit_pending = False

            for bar in future_opt:
                bar_time = bar["time"]

                # 1. Pending ST flip from previous bar → exit at this bar's open
                if st_exit_pending:
                    exit_raw = float(bar["open"])
                    exit_time = bar_time
                    exit_reason = "EOD" if bar_time >= eod_exit else "ST_FLIP"
                    break

                # 2. Reached EOD → exit at this bar's open
                if bar_time >= eod_exit:
                    exit_reason = "EOD"
                    exit_raw = float(bar["open"])
                    exit_time = bar_time
                    break

                # 3. SL / TGT hit during this bar
                reason, level = choose_exit_reason_for_short(
                    bar, sl_price, tgt_price
                )
                if reason:
                    exit_reason = reason
                    exit_raw = float(level)
                    exit_time = bar_time
                    break

                # Track last bar as fallback (in case loop ends without break)
                exit_raw = float(bar["close"])
                exit_time = bar_time

                # 4. ST flip check at end of this closed candle
                if st_exit_mode != "none":
                    s_st = spot_st_map.get(bar_time)
                    if s_st is not None:
                        spot_against = (
                            (direction == "Short" and s_st == 1)
                            or (direction == "Long" and s_st == -1)
                        )
                        if spot_against:
                            if st_exit_mode in ("spot_only", "reverse"):
                                st_exit_pending = True
                            elif st_exit_mode == "spot_and_option":
                                o_st = opt_st_map.get(bar_time)
                                if o_st is not None and o_st == 1:
                                    st_exit_pending = True

            # ── P&L ──
            raw_exit, raw_pnl = shared_engine.compute_realized_pnl(
                entry_price, exit_raw, "sell", slippage
            )
            pnl_pts = raw_pnl - brokerage_pts
            pnl_inr = pnl_pts * lot_size * num_lots

            pivot_name, pivot_value = touched_levels[0]

            trades.append(
                {
                    "date": trade_date,
                    "day_of_week": trade_date.strftime("%A"),
                    "direction": direction,
                    "position_side": "short_option",
                    "trade_mode": "sell",
                    "opt_type": opt_type,
                    "strike": strike,
                    "atm": shared_engine.get_strike(candle["close"], 0, 50),
                    "signal_time": signal_time,
                    "trigger_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "or_high": None,
                    "or_low": None,
                    "spot_sl": None,
                    "spot_at_signal": round(candle["close"], 2),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_raw, 2),
                    "target_price": round(tgt_price, 2),
                    "sl_level": round(sl_price, 2),
                    "exit_reason": exit_reason,
                    "pnl_pts": round(pnl_pts, 2),
                    "pnl_inr": round(pnl_inr, 2),
                    "pivot_name": pivot_name,
                    "pivot_value": pivot_value,
                    "signal_open": round(candle["open"], 2),
                    "signal_high": round(candle["high"], 2),
                    "signal_low": round(candle["low"], 2),
                    "signal_close": round(candle["close"], 2),
                    "prev_day_close": round(
                        float(prev_day_spot["close"].iloc[-1]), 2
                    ),
                    "touched_pivots": ",".join(
                        name for name, _ in touched_levels
                    ),
                    "filled_exit_price": round(raw_exit, 2),
                    "expiry_code": expiry_code,
                    "pivots": pivots,
                    "spot_st_at_signal": st_dir,
                }
            )

            # ── Re-entry: opposite trade after ST flip exit ──
            if (
                st_exit_mode == "reverse"
                and exit_reason == "ST_FLIP"
                and exit_time < eod_exit
            ):
                re_opt_type = "PE" if opt_type == "CE" else "CE"
                re_direction = "Long" if re_opt_type == "PE" else "Short"

                # Spot price from the ST flip trigger candle (one bar before exit)
                trigger_time_rev = shift_time_by_minutes(
                    exit_time, -candle_minutes
                )
                spot_at_rev = current_spot[
                    current_spot["time"] == trigger_time_rev
                ]
                if spot_at_rev.empty:
                    spot_at_rev = current_spot[
                        current_spot["time"] < exit_time
                    ].tail(1)
                if not spot_at_rev.empty:
                    re_spot_price = float(spot_at_rev.iloc[0]["close"])
                    re_strike = calculate_strike(re_spot_price, re_opt_type)

                    re_opt_raw = current_opts[
                        (current_opts["actual_strike"] == re_strike)
                        & (current_opts["option_type"] == re_opt_type)
                    ].sort_values("datetime_ist")

                    if not re_opt_raw.empty:
                        re_opt_5m = resample_to_minutes(
                            re_opt_raw, candle_minutes
                        )
                        re_entry_cands = re_opt_5m[
                            re_opt_5m["time"] >= exit_time
                        ]

                        if not re_entry_cands.empty:
                            re_bar0 = re_entry_cands.iloc[0]
                            re_raw_entry = float(re_bar0["open"])
                            re_entry_time = re_bar0["time"]
                            re_entry_price = shared_engine.apply_slippage(
                                re_raw_entry, "entry_sell", slippage
                            )

                            # SL / TGT (None → use inf so check never triggers)
                            re_sl = (
                                re_entry_price + reentry_sl_pts
                                if reentry_sl_pts is not None
                                else float("inf")
                            )
                            re_tgt = (
                                re_entry_price - reentry_tgt_pts
                                if reentry_tgt_pts is not None
                                else float("-inf")
                            )

                            re_future = re_opt_5m[
                                re_opt_5m["time"] > re_entry_time
                            ].to_dict("records")
                            re_exit_reason = "EOD"
                            re_exit_time = re_entry_time
                            re_exit_raw = re_raw_entry

                            for rbar in re_future:
                                rbar_time = rbar["time"]

                                if rbar_time >= eod_exit:
                                    re_exit_reason = "EOD"
                                    re_exit_raw = float(rbar["open"])
                                    re_exit_time = rbar_time
                                    break

                                reason_r, level_r = (
                                    choose_exit_reason_for_short(
                                        rbar, re_sl, re_tgt
                                    )
                                )
                                if reason_r:
                                    re_exit_reason = reason_r
                                    re_exit_raw = float(level_r)
                                    re_exit_time = rbar_time
                                    break

                                re_exit_raw = float(rbar["close"])
                                re_exit_time = rbar_time

                            re_raw_exit, re_raw_pnl = (
                                shared_engine.compute_realized_pnl(
                                    re_entry_price, re_exit_raw,
                                    "sell", slippage,
                                )
                            )
                            re_pnl_pts = re_raw_pnl - brokerage_pts
                            re_pnl_inr = re_pnl_pts * lot_size * num_lots

                            trades.append(
                                {
                                    "date": trade_date,
                                    "day_of_week": trade_date.strftime("%A"),
                                    "direction": re_direction,
                                    "position_side": "short_option",
                                    "trade_mode": "sell",
                                    "opt_type": re_opt_type,
                                    "strike": re_strike,
                                    "atm": shared_engine.get_strike(
                                        re_spot_price, 0, 50
                                    ),
                                    "signal_time": exit_time,
                                    "trigger_time": exit_time,
                                    "entry_time": re_entry_time,
                                    "exit_time": re_exit_time,
                                    "or_high": None,
                                    "or_low": None,
                                    "spot_sl": None,
                                    "spot_at_signal": round(
                                        re_spot_price, 2
                                    ),
                                    "entry_price": round(
                                        re_entry_price, 2
                                    ),
                                    "exit_price": round(re_exit_raw, 2),
                                    "target_price": (
                                        round(re_tgt, 2)
                                        if reentry_tgt_pts is not None
                                        else None
                                    ),
                                    "sl_level": (
                                        round(re_sl, 2)
                                        if reentry_sl_pts is not None
                                        else None
                                    ),
                                    "exit_reason": re_exit_reason,
                                    "pnl_pts": round(re_pnl_pts, 2),
                                    "pnl_inr": round(re_pnl_inr, 2),
                                    "pivot_name": pivot_name,
                                    "pivot_value": pivot_value,
                                    "signal_open": None,
                                    "signal_high": None,
                                    "signal_low": None,
                                    "signal_close": None,
                                    "prev_day_close": round(
                                        float(
                                            prev_day_spot["close"].iloc[-1]
                                        ),
                                        2,
                                    ),
                                    "touched_pivots": "",
                                    "filled_exit_price": round(
                                        re_raw_exit, 2
                                    ),
                                    "expiry_code": expiry_code,
                                    "pivots": pivots,
                                    "spot_st_at_signal": -st_dir,
                                }
                            )

            trade_taken = True

    return pd.DataFrame(trades)


def main() -> None:
    cfg = get_config(DEFAULT_CONFIG)
    trades_df = build_trade_rows(cfg)

    if trades_df.empty:
        stats = {
            "message": "No Pivot+ST Sell trades were generated for the selected range.",
            "total_trades": 0,
        }
        print(stats["message"])
        save_results(trades_df, stats, cfg)
        return

    stats = shared_engine.compute_stats(trades_df, cfg)
    stats["strategy"] = "Pivot + ST Sell"
    stats["notes"] = (
        "Pivot touch + Supertrend direction for entry (ST bearish → sell CE, "
        "ST bullish → sell PE). ST flip exit when spot ST reverses AND "
        "option ST turns bullish. SL/TGT/EOD exits also apply."
    )

    print(
        f"Pivot+ST Sell backtest complete: {len(trades_df)} trades | "
        f"PnL {stats.get('total_pnl_pts', 0)} pts"
    )
    save_results(trades_df, stats, cfg)


if __name__ == "__main__":
    main()
