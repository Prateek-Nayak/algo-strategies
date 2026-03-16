from __future__ import annotations

from collections import deque
from datetime import datetime, time, timedelta
from pathlib import Path
import math
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
    "DATE_FROM": "2025-01-01",
    "DATE_TO": "2025-12-31",
    "SPOT_CANDLE_MINUTES": 5,
    "ENTRY_START_TIME": "10:00",
    "LAST_SIGNAL_TIME": "14:59",
    "EOD_EXIT_TIME": "15:00",
    "EXPIRY_CODE": 1,
    "EXPIRY_CODE_ON_EXPIRY_DAY": 2,  # 2 = next week (avoid 0DTE), 1 = use 0DTE
    "LOT_SIZE": 75,
    "NUM_LOTS": 1,
    "TARGET_POINTS": 50,
    "STOP_LOSS_POINTS": 50,
    "BROKERAGE_PER_TRADE": 40,
    "SLIPPAGE_POINTS": 0.5,
    "ST_PERIOD": 10,
    "ST_MULTIPLIER": 3.0,
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


# ── Core logic (untouched) ──────────────────────────────────────────────────


class SupertrendLive:
    """
    Candle-by-candle Supertrend calculator (TradingView compatible).
    """
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        self.reset()

    def reset(self):
        self._candle_count = 0
        self._prev_close = None
        self._atr = None
        self._tr_buffer = deque(maxlen=self.period)
        self._final_ub = None
        self._final_lb = None
        self._trend = True
        self._supertrend = None

    @property
    def is_ready(self):
        return self._supertrend is not None

    def current(self):
        if not self.is_ready:
            return None, None
        return (1 if self._trend else -1), self._supertrend

    def update(self, o, h, l, c):
        self._candle_count += 1

        if self._prev_close is not None:
            tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        else:
            tr = h - l

        self._tr_buffer.append(tr)

        if self._candle_count < self.period:
            self._prev_close = c
            return None, None

        if self._candle_count == self.period:
            self._atr = sum(self._tr_buffer) / self.period
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        hl2 = (h + l) / 2
        basic_ub = hl2 + self.multiplier * self._atr
        basic_lb = hl2 - self.multiplier * self._atr

        if self._final_ub is None:
            self._final_ub = basic_ub
            self._final_lb = basic_lb
            self._supertrend = basic_ub
            self._trend = True
        else:
            if basic_ub < self._final_ub or self._prev_close > self._final_ub:
                new_ub = basic_ub
            else:
                new_ub = self._final_ub

            if basic_lb > self._final_lb or self._prev_close < self._final_lb:
                new_lb = basic_lb
            else:
                new_lb = self._final_lb

            self._final_ub = new_ub
            self._final_lb = new_lb

            if self._trend:
                if c <= self._final_lb:
                    self._trend = False
                    self._supertrend = self._final_ub
                else:
                    self._trend = True
                    self._supertrend = self._final_lb
            else:
                if c >= self._final_ub:
                    self._trend = True
                    self._supertrend = self._final_lb
                else:
                    self._trend = False
                    self._supertrend = self._final_ub

        self._prev_close = c
        direction = 1 if self._trend else -1
        return direction, round(self._supertrend, 2)


def compute_pivots(high, low, close):
    """Compute classic pivot levels: P, R1-R10, S1-S10."""
    P = (high + low + close) / 3
    pivots = {"P": round(P, 2)}

    R1 = 2 * P - low
    S1 = 2 * P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    pivots["R1"] = round(R1, 2)
    pivots["R2"] = round(R2, 2)
    pivots["S1"] = round(S1, 2)
    pivots["S2"] = round(S2, 2)

    for n in range(3, 11):
        Rn = high + (n - 1) * (P - low)
        Sn = low - (n - 1) * (high - P)
        pivots[f"R{n}"] = round(Rn, 2)
        pivots[f"S{n}"] = round(Sn, 2)

    return pivots


def calculate_strike_price(close, direction):
    if direction == "CE":
        return int(math.floor(close / 50) * 50)
    elif direction == "PE":
        return int(math.ceil(close / 50) * 50)


def find_triggered_pivot(candle_high, candle_low, pivot_levels):
    for name, level in pivot_levels.items():
        if candle_low <= level <= candle_high:
            return name, level
    return None, None


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

    # ── Supertrend on full spot (cross-day for natural warm-up) ──
    st_period = int(cfg["ST_PERIOD"])
    st_multiplier = float(cfg["ST_MULTIPLIER"])

    st_calc = SupertrendLive(period=st_period, multiplier=st_multiplier)
    st_dirs = []
    for _, row in spot_5m.iterrows():
        d, _ = st_calc.update(row["open"], row["high"], row["low"], row["close"])
        st_dirs.append(d)
    spot_5m = spot_5m.copy()
    spot_5m["st_dir"] = st_dirs

    # ── Config values ──
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
    expiry_code_on_expiry = cfg.get("EXPIRY_CODE_ON_EXPIRY_DAY", 2)
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
        expiry_code = (
            expiry_code_on_expiry if trade_date in expiry_dates
            else default_expiry_code
        )
        current_opts = (
            opts[
                (opts["date"] == trade_date)
                & (opts["expiry_code"] == expiry_code)
            ]
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

        # ── Walk spot candles: entry signal + ST flip exit ──
        pending_trigger = None
        entry = None
        exit_info = None
        st_exit_pending = False

        for _, bar in current_spot.iterrows():
            bar_time = bar["time"]
            in_trade_window = entry_start <= bar_time < last_signal

            # 1. Handle st_exit_pending → exit at THIS bar's open
            if entry is not None and st_exit_pending and exit_info is None:
                if bar_time >= eod_exit:
                    exit_info = {"time": bar_time, "reason": "EOD"}
                else:
                    exit_info = {"time": bar_time, "reason": "ST_FLIP"}
                break

            # 2. EOD check
            if bar_time >= eod_exit:
                if entry is not None and exit_info is None:
                    exit_info = {"time": bar_time, "reason": "EOD"}
                break

            # 3. Act on pending pivot trigger → enter at this bar
            if pending_trigger is not None and entry is None:
                st_dir = bar.get("st_dir")
                if st_dir is not None and not (
                    isinstance(st_dir, float) and np.isnan(st_dir)
                ):
                    direction = "PE" if st_dir == 1 else "CE"
                    strike = calculate_strike_price(
                        pending_trigger["close"], direction
                    )
                    entry = {
                        "time": bar_time,
                        "nifty_price": float(bar["open"]),
                        "st_dir": int(st_dir),
                        "direction": direction,
                        "strike": strike,
                        "pivot_name": pending_trigger["pivot_name"],
                        "pivot_level": pending_trigger["pivot_level"],
                        "trigger_close": pending_trigger["close"],
                        "trigger_time": pending_trigger["time"],
                    }
                pending_trigger = None
                continue

            # 4. New pivot trigger (only in trade window, no existing entry)
            if entry is None and pending_trigger is None and in_trade_window:
                pname, plevel = find_triggered_pivot(
                    float(bar["high"]), float(bar["low"]), pivots
                )
                if pname is not None:
                    pending_trigger = {
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "pivot_name": pname,
                        "pivot_level": plevel,
                        "time": bar_time,
                    }
                continue

            # 5. ST flip check (after entry, sets pending for next bar)
            if entry is not None and exit_info is None and not st_exit_pending:
                st_dir = bar.get("st_dir")
                if st_dir is not None and not (
                    isinstance(st_dir, float) and np.isnan(st_dir)
                ):
                    if (
                        (entry["direction"] == "CE" and st_dir == 1)
                        or (entry["direction"] == "PE" and st_dir == -1)
                    ):
                        st_exit_pending = True

        # EOD fallback: if entry exists but no exit after loop
        if entry is not None and exit_info is None:
            last_bar = current_spot.iloc[-1]
            exit_info = {"time": last_bar["time"], "reason": "EOD"}

        if entry is None:
            continue

        # ── Option data: entry premium + TP/SL scan (1-min) ──
        opt_day = current_opts[
            (current_opts["actual_strike"] == float(entry["strike"]))
            & (current_opts["option_type"] == entry["direction"])
        ].sort_values("time")

        if opt_day.empty:
            continue

        entry_mask = opt_day["time"] >= entry["time"]
        if not entry_mask.any():
            continue

        entry_bar = opt_day[entry_mask].iloc[0]
        raw_entry = float(entry_bar["open"])
        option_entry_time = entry_bar["time"]

        if raw_entry <= 0:
            continue

        entry_price = shared_engine.apply_slippage(
            raw_entry, "entry_sell", slippage
        )

        sl_level = entry_price + stop_loss_pts
        tp_level = entry_price - target_pts

        # Walk 1-min option candles for TP/SL
        tp_sl_exit = None
        scan_bars = opt_day[opt_day["time"] > option_entry_time]
        for _, orow in scan_bars.iterrows():
            # SL first: premium rises to sl_level
            if orow["high"] >= sl_level:
                tp_sl_exit = {
                    "time": orow["time"],
                    "premium": sl_level,
                    "reason": "OPT_SL",
                }
                break
            # TP: premium drops to tp_level
            if orow["low"] <= tp_level:
                tp_sl_exit = {
                    "time": orow["time"],
                    "premium": tp_level,
                    "reason": "TARGET",
                }
                break

        # ── Determine final exit ──
        if tp_sl_exit and (
            exit_info is None or tp_sl_exit["time"] <= exit_info["time"]
        ):
            exit_reason = tp_sl_exit["reason"]
            exit_raw = tp_sl_exit["premium"]
            final_exit_time = tp_sl_exit["time"]
        else:
            exit_reason = exit_info["reason"]
            exit_opt_mask = opt_day["time"] >= exit_info["time"]
            if exit_opt_mask.any():
                exit_raw = float(opt_day[exit_opt_mask].iloc[0]["open"])
                final_exit_time = opt_day[exit_opt_mask].iloc[0]["time"]
            else:
                exit_raw = float(opt_day.iloc[-1]["close"])
                final_exit_time = opt_day.iloc[-1]["time"]

        # ── P&L ──
        raw_exit, raw_pnl = shared_engine.compute_realized_pnl(
            entry_price, exit_raw, "sell", slippage
        )

        # Cap PnL to TP/SL boundaries for non-TP/SL exits
        if exit_reason not in ("OPT_SL", "TARGET"):
            raw_pnl = max(-stop_loss_pts, min(target_pts, raw_pnl))

        pnl_pts = raw_pnl - brokerage_pts
        pnl_inr = pnl_pts * lot_size * num_lots

        direction = "Long" if entry["direction"] == "PE" else "Short"
        pivot_name = entry["pivot_name"]
        pivot_value = entry["pivot_level"]

        trades.append(
            {
                "date": trade_date,
                "day_of_week": trade_date.strftime("%A"),
                "direction": direction,
                "position_side": "short_option",
                "trade_mode": "sell",
                "opt_type": entry["direction"],
                "strike": entry["strike"],
                "atm": shared_engine.get_strike(
                    entry["nifty_price"], 0, 50
                ),
                "signal_time": entry["trigger_time"],
                "trigger_time": entry["trigger_time"],
                "entry_time": option_entry_time,
                "exit_time": final_exit_time,
                "or_high": None,
                "or_low": None,
                "spot_sl": None,
                "spot_at_signal": round(entry["nifty_price"], 2),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_raw, 2),
                "target_price": round(tp_level, 2),
                "sl_level": round(sl_level, 2),
                "exit_reason": exit_reason,
                "pnl_pts": round(pnl_pts, 2),
                "pnl_inr": round(pnl_inr, 2),
                "pivot_name": pivot_name,
                "pivot_value": pivot_value,
                "prev_day_close": round(
                    float(prev_day_spot["close"].iloc[-1]), 2
                ),
                "filled_exit_price": round(raw_exit, 2),
                "expiry_code": expiry_code,
                "pivots": pivots,
                "spot_st_at_signal": entry["st_dir"],
            }
        )

    return pd.DataFrame(trades)


def main() -> None:
    cfg = get_config(DEFAULT_CONFIG)
    trades_df = build_trade_rows(cfg)

    if trades_df.empty:
        stats = {
            "message": "No Pivot+ST Fliptrade trades were generated for the selected range.",
            "total_trades": 0,
        }
        print(stats["message"])
        save_results(trades_df, stats, cfg)
        return

    stats = shared_engine.compute_stats(trades_df, cfg)
    stats["strategy"] = "Pivot + ST Fliptrade"
    stats["notes"] = (
        "Pivot touch pending trigger + Supertrend direction for entry "
        "(ST bearish → sell CE, ST bullish → sell PE). "
        "Exits: option TP/SL (±50pts), Nifty ST flip (next bar open), EOD."
    )

    print(
        f"Pivot+ST Fliptrade backtest complete: {len(trades_df)} trades | "
        f"PnL {stats.get('total_pnl_pts', 0)} pts"
    )
    save_results(trades_df, stats, cfg)


if __name__ == "__main__":
    main()
