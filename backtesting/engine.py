"""
Backtesting engine — reusable core extracted from orb_backtest_single.py.
Exposes: load_data, run_backtest, compute_stats
"""

import os
import sys
import glob
import warnings
from datetime import time, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
ALLOWED_DIRECTIONS = {"both", "long_only", "short_only"}
ALLOWED_TRADE_MODES = {"buy", "sell"}
ALLOWED_SPOT_SL_MODES = {"breakout_level", "opposite"}
SPOT_LOAD_COLUMNS = ["date", "open", "high", "low", "close"]
OPTION_LOAD_COLUMNS = [
    "datetime_ist",
    "open",
    "high",
    "low",
    "close",
    "actual_strike",
    "expiry_code",
    "option_type",
]


# =============================================================================
# HELPERS
# =============================================================================
def parse_time(t_str):
    h, m = map(int, t_str.split(":"))
    return time(h, m)


def get_strike(spot_price, offset, interval):
    atm = int(round(spot_price / interval) * interval)
    return atm + offset * interval


def apply_slippage(price, side, slippage):
    return price + slippage if side in ("entry_buy", "close_short") else price - slippage


def brokerage_in_pts(lot_size, num_lots, brokerage_inr):
    return brokerage_inr / (lot_size * num_lots)


def shift_time_by_minutes(t_value, minutes):
    base = datetime.combine(datetime(2000, 1, 1).date(), t_value)
    return (base + timedelta(minutes=minutes)).time()


def resolve_spot_sl_exit(option_bar):
    """
    Exit at the current option candle close. This function is called on the
    option bar aligned with the 1-min spot candle whose high/low touched the SL.
    """
    return option_bar["close"], option_bar["time"]


def normalize_trade_mode(value):
    trade_mode = str(value).strip().lower()
    if trade_mode not in ALLOWED_TRADE_MODES:
        raise ValueError(f"TRADE_MODE must be one of {sorted(ALLOWED_TRADE_MODES)}")
    return trade_mode


def normalize_direction(value):
    direction = str(value).strip().lower()
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError(f"DIRECTION must be one of {sorted(ALLOWED_DIRECTIONS)}")
    return direction


def normalize_spot_sl_mode(value):
    mode = str(value).strip().lower()
    if mode not in ALLOWED_SPOT_SL_MODES:
        raise ValueError(f"SPOT_SL_MODE must be one of {sorted(ALLOWED_SPOT_SL_MODES)}")
    return mode


def validate_config(cfg):
    parse_ymd_date(cfg["DATE_FROM"])
    parse_ymd_date(cfg["DATE_TO"])
    if cfg["DATE_FROM"] > cfg["DATE_TO"]:
        raise ValueError("DATE_FROM must be on or before DATE_TO")

    parse_time(cfg["ORB_START"])
    parse_time(cfg["ORB_WINDOW_END"])
    parse_time(cfg["EOD_EXIT_TIME"])

    if cfg.get("SPOT_CANDLE_MINUTES", 1) <= 0:
        raise ValueError("SPOT_CANDLE_MINUTES must be positive")
    if cfg["STRIKE_INTERVAL"] <= 0:
        raise ValueError("STRIKE_INTERVAL must be positive")
    if cfg["LOT_SIZE"] <= 0 or cfg["NUM_LOTS"] <= 0:
        raise ValueError("LOT_SIZE and NUM_LOTS must be positive")
    if cfg["BROKERAGE_PER_TRADE"] < 0 or cfg["SLIPPAGE_POINTS"] < 0:
        raise ValueError("BROKERAGE_PER_TRADE and SLIPPAGE_POINTS cannot be negative")

    for key in ("OPTION_SL_PTS", "OPTION_TGT_PTS"):
        value = cfg.get(key)
        if value is not None and value < 0:
            raise ValueError(f"{key} cannot be negative")

    normalize_trade_mode(cfg["TRADE_MODE"])
    normalize_direction(cfg["DIRECTION"])
    normalize_spot_sl_mode(cfg["SPOT_SL_MODE"])


def option_position_side(trade_mode):
    return "long_option" if trade_mode == "buy" else "short_option"


def resolve_option_type(signal_direction, trade_mode):
    if trade_mode == "buy":
        return "CE" if signal_direction == "Long" else "PE"
    return "PE" if signal_direction == "Long" else "CE"


def build_option_levels(entry_price, trade_mode, sl_pts, tgt_pts):
    if trade_mode == "buy":
        sl_price = (entry_price - sl_pts) if sl_pts is not None else None
        tgt_price = (entry_price + tgt_pts) if tgt_pts is not None else None
    else:
        sl_price = (entry_price + sl_pts) if sl_pts is not None else None
        tgt_price = (entry_price - tgt_pts) if tgt_pts is not None else None
    return sl_price, tgt_price


def option_level_hits(option_bar, trade_mode, sl_price, tgt_price):
    if trade_mode == "buy":
        opt_sl_hit = sl_price is not None and option_bar["low"] <= sl_price
        opt_tgt_hit = tgt_price is not None and option_bar["high"] >= tgt_price
    else:
        opt_sl_hit = sl_price is not None and option_bar["high"] >= sl_price
        opt_tgt_hit = tgt_price is not None and option_bar["low"] <= tgt_price
    return opt_sl_hit, opt_tgt_hit


def compute_realized_pnl(entry_price, exit_price, trade_mode, slippage):
    if trade_mode == "buy":
        raw_exit = apply_slippage(exit_price, "close_long", slippage)
        raw_pnl = raw_exit - entry_price
    else:
        raw_exit = apply_slippage(exit_price, "close_short", slippage)
        raw_pnl = entry_price - raw_exit
    return raw_exit, raw_pnl


def spot_sl_touched(spot_bar, signal_direction, spot_sl):
    if signal_direction == "Long":
        return spot_bar["low"] <= spot_sl
    return spot_bar["high"] >= spot_sl


# =============================================================================
# DATA LOADING
# =============================================================================
def parse_ymd_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


def month_end(ms):
    return (ms.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)


def dataset_file_matches_date_range(fp, date_from=None, date_to=None):
    sd, ed = parse_ymd_date(date_from), parse_ymd_date(date_to)
    stem = os.path.splitext(os.path.basename(fp))[0]
    try:
        if len(stem) == 10:
            fs = datetime.strptime(stem, "%Y-%m-%d").date()
            fe = fs
        elif len(stem) == 7:
            fs = datetime.strptime(stem, "%Y-%m").date()
            fe = month_end(fs)
        else:
            return True
    except ValueError:
        return True
    if sd and fe < sd:
        return False
    if ed and fs > ed:
        return False
    return True


def normalize_loaded_data(df, path):
    if "datetime_ist" in df.columns:
        df["datetime_ist"] = pd.to_datetime(df["datetime_ist"], utc=True).dt.tz_convert(
            "Asia/Kolkata"
        )
    elif "date" in df.columns:
        df["datetime_ist"] = pd.to_datetime(df["date"], dayfirst=True).dt.tz_localize(
            "Asia/Kolkata"
        )
        df = df.drop(columns=["date"])
    else:
        raise ValueError(f"No datetime column in {path}")
    df = df.sort_values("datetime_ist").reset_index(drop=True)
    df["date"] = df["datetime_ist"].dt.date
    df["time"] = df["datetime_ist"].dt.time
    return df


def load_dataset_folder(path, date_from=None, date_to=None, label="data", columns=None):
    pq = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    cs = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    ft, files = ("parquet", pq) if pq else ("csv", cs) if cs else (None, [])
    if not ft:
        raise FileNotFoundError(f"No files under {path}")
    files = [f for f in files if dataset_file_matches_date_range(f, date_from, date_to)]
    if not files:
        raise FileNotFoundError(f"No {ft} files for {date_from}-{date_to}")
    dfs = []
    for f in files:
        try:
            if ft == "parquet":
                dfs.append(pd.read_parquet(f, columns=columns))
            elif columns:
                dfs.append(pd.read_csv(f, usecols=lambda col: col in columns))
            else:
                dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Warn: {f} — {e}")
    df = pd.concat(dfs, ignore_index=True)
    df = normalize_loaded_data(df, path)
    if date_from:
        df = df[df["date"] >= datetime.strptime(date_from, "%Y-%m-%d").date()]
    if date_to:
        df = df[df["date"] <= datetime.strptime(date_to, "%Y-%m-%d").date()]
    df = df.reset_index(drop=True)
    print(f"  {label:<8}: {len(df):>9,} rows | {df['date'].nunique():>4} days ({ft})")
    return df


def resample_spot(df, minutes):
    if minutes == 1:
        return df

    def _rs(g):
        r = (
            g.set_index("datetime_ist")
            .resample(f"{minutes}min", closed="left", label="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna(subset=["open"])
        )
        return r

    out = df.groupby("date", group_keys=False).apply(_rs).reset_index()
    out["date"] = out["datetime_ist"].dt.date
    out["time"] = out["datetime_ist"].dt.time
    return out.sort_values("datetime_ist").reset_index(drop=True)


def load_data(cfg):
    """Load spot and options data. Paths can be absolute or relative to BASE_DIR."""
    validate_config(cfg)
    spot_path = cfg["SPOT_DATA_PATH"]
    opt_path = cfg["OPT_DATA_PATH"]

    # Resolve relative paths
    if not os.path.isabs(spot_path):
        spot_path = str(BASE_DIR / spot_path)
    if not os.path.isabs(opt_path):
        opt_path = str(BASE_DIR / opt_path)

    spot_raw = load_dataset_folder(
        spot_path,
        cfg["DATE_FROM"],
        cfg["DATE_TO"],
        "Spot",
        columns=SPOT_LOAD_COLUMNS,
    )
    opts = load_dataset_folder(
        opt_path,
        cfg["DATE_FROM"],
        cfg["DATE_TO"],
        "Options",
        columns=OPTION_LOAD_COLUMNS,
    )
    opts = opts[opts["expiry_code"] == cfg["EXPIRY_CODE"]].reset_index(drop=True)
    print(f"  Options after expiry filter (code={cfg['EXPIRY_CODE']}): {len(opts):,} rows")
    mins = cfg.get("SPOT_CANDLE_MINUTES", 1)
    if mins > 1:
        spot = resample_spot(spot_raw, mins)
        print(
            f"  Spot resampled to {mins}-min: {len(spot):,} bars | {spot['date'].nunique()} days"
        )
    else:
        spot = spot_raw
    return spot, spot_raw, opts


# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def run_backtest(spot, spot_raw, opts, cfg):
    validate_config(cfg)
    window_end = parse_time(cfg["ORB_WINDOW_END"])
    eod_time = parse_time(cfg["EOD_EXIT_TIME"])
    trade_mode = normalize_trade_mode(cfg["TRADE_MODE"])
    direction = normalize_direction(cfg["DIRECTION"])
    strike_off = cfg["STRIKE_OFFSET"]
    interval = cfg["STRIKE_INTERVAL"]
    lot_size = cfg["LOT_SIZE"]
    num_lots = cfg["NUM_LOTS"]
    slippage = cfg["SLIPPAGE_POINTS"]
    use_spot_sl = cfg["USE_SPOT_SL"]
    spot_sl_mode = normalize_spot_sl_mode(cfg["SPOT_SL_MODE"])
    spot_candle_minutes = cfg.get("SPOT_CANDLE_MINUTES", 1)
    sl_pts = cfg["OPTION_SL_PTS"]
    tgt_pts = cfg["OPTION_TGT_PTS"]
    brk_pts = brokerage_in_pts(lot_size, num_lots, cfg["BROKERAGE_PER_TRADE"])

    trades = []
    for trade_date, day_spot in spot.groupby("date"):
        day_spot = day_spot.sort_values("time").reset_index(drop=True)
        day_opts = opts[opts["date"] == trade_date].copy()
        if day_opts.empty:
            continue

        or_bars = day_spot[day_spot["time"] < window_end]
        # Spot bars are left-labeled after resampling, so a 09:45 bar covers
        # 09:45-09:50 and closes at 09:50. That first post-ORB bar must be
        # eligible to trigger an entry.
        post_bars = day_spot[
            (day_spot["time"] >= window_end) & (day_spot["time"] <= eod_time)
        ]
        if len(or_bars) < 1 or len(post_bars) < 1:
            continue

        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()

        trade_taken = False
        for _, bar in post_bars.iterrows():
            if trade_taken:
                break
            sig_dir = None
            if bar["close"] > or_high and direction in ("both", "long_only"):
                sig_dir = "Long"
            elif bar["close"] < or_low and direction in ("both", "short_only"):
                sig_dir = "Short"
            if sig_dir is None:
                continue
            trade_taken = True

            signal_time = bar["time"]
            # Entry after the 5-min candle closes (e.g. 10:00 candle → enter at 10:05)
            signal_close_time = shift_time_by_minutes(signal_time, spot_candle_minutes)
            spot_price = bar["close"]
            opt_type = resolve_option_type(sig_dir, trade_mode)

            if spot_sl_mode == "breakout_level":
                spot_sl = or_high if sig_dir == "Long" else or_low
            else:
                spot_sl = or_low if sig_dir == "Long" else or_high

            eff_offset = strike_off if opt_type == "CE" else -strike_off
            strike = get_strike(spot_price, eff_offset, interval)

            day_opt_f = day_opts[
                (day_opts["actual_strike"] == strike)
                & (day_opts["option_type"] == opt_type)
            ].sort_values("time")
            eb = day_opt_f[day_opt_f["time"] >= signal_close_time]
            if eb.empty:
                continue
            raw_entry = eb.iloc[0]["open"]
            entry_time = eb.iloc[0]["time"]
            if raw_entry <= 0:
                continue

            entry_price = apply_slippage(
                raw_entry,
                "entry_buy" if trade_mode == "buy" else "entry_sell",
                slippage,
            )

            # SL / Target levels
            sl_price, tgt_price = build_option_levels(entry_price, trade_mode, sl_pts, tgt_pts)

            # Walk forward
            post_opt = day_opt_f[
                (day_opt_f["time"] > entry_time) & (day_opt_f["time"] <= eod_time)
            ].reset_index(drop=True)
            # Use 1-min spot data for exit scanning (touch detection)
            day_spot_raw = spot_raw[spot_raw["date"] == trade_date].sort_values("time")
            post_spot_1m = day_spot_raw[
                (day_spot_raw["time"] > entry_time) & (day_spot_raw["time"] <= eod_time)
            ]

            exit_price = None
            exit_reason = "EOD"
            exit_time = None

            spot_idx = 0
            post_spot_list = post_spot_1m.to_dict("records")
            post_opt_list = post_opt.to_dict("records")

            for obar_idx, obar in enumerate(post_opt_list):
                curr_t = obar["time"]
                opt_close = obar["close"]
                opt_high = obar["high"]
                opt_low_b = obar["low"]
                spot_sl_hit = False
                while (
                    use_spot_sl
                    and spot_idx < len(post_spot_list)
                    and shift_time_by_minutes(
                        post_spot_list[spot_idx]["time"], 1
                    )
                    <= curr_t
                ):
                    spot_bar = post_spot_list[spot_idx]
                    if spot_sl_touched(spot_bar, sig_dir, spot_sl):
                        spot_sl_hit = True
                    spot_idx += 1
                while (
                    not use_spot_sl
                    and spot_idx < len(post_spot_list)
                    and shift_time_by_minutes(
                        post_spot_list[spot_idx]["time"], 1
                    )
                    <= curr_t
                ):
                    spot_idx += 1

                if spot_sl_hit:
                    exit_price, exit_time = resolve_spot_sl_exit(obar)
                    exit_reason = "SPOT_SL"
                    break

                opt_sl_hit, opt_tgt_hit = option_level_hits(obar, trade_mode, sl_price, tgt_price)

                if opt_tgt_hit:
                    exit_price = tgt_price
                    exit_reason = "TARGET"
                    exit_time = curr_t
                    break
                elif opt_sl_hit:
                    exit_price = sl_price
                    exit_reason = "OPT_SL"
                    exit_time = curr_t
                    break
                else:
                    exit_price = opt_close
                    exit_time = curr_t

            if exit_price is None:
                exit_price = entry_price
                exit_reason = "NO_DATA"

            # P&L
            raw_exit, raw_pnl = compute_realized_pnl(entry_price, exit_price, trade_mode, slippage)
            pnl_pts = raw_pnl - brk_pts
            pnl_inr = pnl_pts * lot_size * num_lots

            trades.append(
                {
                    "date": trade_date,
                    "day_of_week": trade_date.strftime("%A"),
                    "direction": sig_dir,
                    "position_side": option_position_side(trade_mode),
                    "trade_mode": trade_mode,
                    "opt_type": opt_type,
                    "strike": strike,
                    "atm": get_strike(spot_price, 0, interval),
                    "signal_time": signal_time,
                    "trigger_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "or_high": round(or_high, 2),
                    "or_low": round(or_low, 2),
                    "spot_sl": round(spot_sl, 2),
                    "spot_at_signal": round(spot_price, 2),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "target_price": round(tgt_price, 2) if tgt_price is not None else None,
                    "sl_level": round(sl_price, 2) if sl_price is not None else None,
                    "exit_reason": exit_reason,
                    "pnl_pts": round(pnl_pts, 2),
                    "pnl_inr": round(pnl_inr, 2),
                }
            )

    return pd.DataFrame(trades)


# =============================================================================
# STATISTICS
# =============================================================================
def compute_stats(df, cfg):
    if df.empty:
        return {}
    pnl = df["pnl_pts"]
    cum = pnl.cumsum()
    wins = df[pnl > 0]
    losses = df[pnl <= 0]
    wr = len(wins) / len(df) * 100
    avg_w = wins["pnl_pts"].mean() if len(wins) > 0 else 0
    avg_l = losses["pnl_pts"].mean() if len(losses) > 0 else 0
    rr = abs(avg_w / avg_l) if avg_l != 0 else 0
    gw = wins["pnl_pts"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl_pts"].sum()) if len(losses) > 0 else 0
    pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0)
    run_max = cum.cummax()
    dd = run_max - cum
    mdd = dd.max()

    # Streaks
    flags = (pnl > 0).tolist()
    mws = mls = cw = cl = 0
    for v in flags:
        if v:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        mws = max(mws, cw)
        mls = max(mls, cl)

    long_df = df[df["direction"] == "Long"]
    short_df = df[df["direction"] == "Short"]
    lwr = (
        len(long_df[long_df["pnl_pts"] > 0]) / len(long_df) * 100
        if len(long_df) > 0
        else 0
    )
    swr = (
        len(short_df[short_df["pnl_pts"] > 0]) / len(short_df) * 100
        if len(short_df) > 0
        else 0
    )

    std_pnl = pnl.std(ddof=0) if len(df) > 1 else 0.0

    lot = cfg["LOT_SIZE"] * cfg["NUM_LOTS"]
    return {
        "total_trades": len(df),
        "win_rate": round(wr, 1),
        "total_pnl_pts": round(cum.iloc[-1], 2),
        "total_pnl_inr": round(df["pnl_inr"].sum(), 0),
        "avg_pnl_pts": round(pnl.mean(), 2),
        "avg_win_pts": round(avg_w, 2),
        "avg_loss_pts": round(avg_l, 2),
        "rr_ratio": round(rr, 2),
        "profit_factor": round(pf, 2),
        "max_dd_pts": round(mdd, 2),
        "max_dd_inr": round(mdd * lot, 0),
        "max_win_streak": mws,
        "max_loss_streak": mls,
        "long_trades": len(long_df),
        "short_trades": len(short_df),
        "long_wr": round(lwr, 1),
        "short_wr": round(swr, 1),
        "exit_counts": df["exit_reason"].value_counts().to_dict(),
        "best_trade_pts": round(pnl.max(), 2),
        "worst_trade_pts": round(pnl.min(), 2),
        "median_pnl_pts": round(pnl.median(), 2),
        "std_pnl_pts": round(std_pnl, 2),
        "avg_hold_mins": 0,
    }
