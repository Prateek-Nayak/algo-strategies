from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
import sys

import pandas as pd

BACKTESTING_DIR = Path(__file__).resolve().parents[2]
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

import engine as shared_engine
from strategy_runtime import DATA_DIR, save_results, get_config


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

    fallback = (DATA_DIR.parent / path_text).resolve()
    return fallback


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


def option_type_for_candle(candle: dict[str, float]) -> str:
    return "PE" if candle["close"] > candle["open"] else "CE"


def calculate_strike(spot_close: float, option_type: str) -> int:
    if option_type == "CE":
        return int((spot_close // 50) * 50)
    return int(((spot_close + 49.9999) // 50) * 50)


def shift_time_by_minutes(value: time, minutes: int) -> time:
    base = datetime.combine(datetime(2000, 1, 1).date(), value)
    return (base + timedelta(minutes=minutes)).time()


def detect_expiry_dates(trading_dates):
    """Detect weekly expiry dates from the trading calendar.

    Nifty weekly expiry falls on Thursday. If Thursday is a holiday the
    last trading day before it (within the same week) becomes expiry.
    """
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


def build_trade_rows(cfg: dict) -> pd.DataFrame:
    start_date = parse_date(cfg["DATE_FROM"])
    end_date = parse_date(cfg["DATE_TO"])
    extended_start = (start_date - timedelta(days=10)).strftime("%Y-%m-%d")

    spot_path = str(resolve_path(cfg["SPOT_DATA_PATH"]))
    opt_path = str(resolve_path(cfg["OPT_DATA_PATH"]))

    spot_raw = shared_engine.load_dataset_folder(
        spot_path,
        extended_start,
        cfg["DATE_TO"],
        "Spot",
    )
    opts = shared_engine.load_dataset_folder(
        opt_path,
        extended_start,
        cfg["DATE_TO"],
        "Options",
    )
    spot_5m = shared_engine.resample_spot(spot_raw, cfg["SPOT_CANDLE_MINUTES"])

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

        previous_dates = [value for value in available_dates if value < trade_date]
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
        # Use expiry_code=2 on expiry days to avoid 0DTE contracts
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

            opt_type = option_type_for_candle(candle)
            strike = calculate_strike(candle["close"], opt_type)
            entry_lookup_time = shift_time_by_minutes(
                signal_time, int(cfg["SPOT_CANDLE_MINUTES"])
            )

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

            exit_reason = "EOD"
            exit_time = entry_time
            exit_raw = raw_entry

            if entry_time < eod_exit:
                future_bars = option_day[option_day["time"] > entry_time].to_dict("records")
                eod_bar = None

                for bar in future_bars:
                    bar_time = bar["time"]
                    if bar_time >= eod_exit:
                        eod_bar = bar
                        break

                    reason, level = choose_exit_reason_for_short(bar, sl_price, tgt_price)
                    if reason:
                        exit_reason = reason
                        exit_raw = float(level)
                        exit_time = bar_time
                        break
                else:
                    pass

                if exit_reason == "EOD":
                    if eod_bar is None:
                        later = option_day[option_day["time"] >= eod_exit]
                        if not later.empty:
                            eod_bar = later.iloc[0].to_dict()
                    if eod_bar is not None:
                        exit_raw = float(eod_bar["open"])
                        exit_time = eod_bar["time"]

            raw_exit, raw_pnl = shared_engine.compute_realized_pnl(
                entry_price, exit_raw, "sell", slippage
            )
            pnl_pts = raw_pnl - brokerage_pts
            pnl_inr = pnl_pts * lot_size * num_lots

            pivot_name, pivot_value = touched_levels[0]
            direction = "Long" if opt_type == "PE" else "Short"

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
                    "prev_day_close": round(float(prev_day_spot["close"].iloc[-1]), 2),
                    "touched_pivots": ",".join(name for name, _ in touched_levels),
                    "filled_exit_price": round(raw_exit, 2),
                    "expiry_code": expiry_code,
                    "pivots": pivots,
                }
            )
            trade_taken = True

    return pd.DataFrame(trades)


def main() -> None:
    cfg = get_config(DEFAULT_CONFIG)
    trades_df = build_trade_rows(cfg)

    if trades_df.empty:
        stats = {
            "message": "No Pivot Sell trades were generated for the selected range.",
            "total_trades": 0,
        }
        print(stats["message"])
        save_results(trades_df, stats, cfg)
        return

    stats = shared_engine.compute_stats(trades_df, cfg)
    stats["strategy"] = "Pivot Sell"
    stats["notes"] = (
        "Short PE on green 5m pivot touch, short CE on red 5m pivot touch, "
        "with 50-point option SL/TGT and 15:00 market-style squareoff."
    )

    print(
        f"Pivot Sell backtest complete: {len(trades_df)} trades | "
        f"PnL {stats.get('total_pnl_pts', 0)} pts"
    )
    save_results(trades_df, stats, cfg)


if __name__ == "__main__":
    main()
