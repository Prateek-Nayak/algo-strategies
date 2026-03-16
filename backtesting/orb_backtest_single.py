"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ORB Options — Single-Config Comprehensive Backtest                    ║
║        Per-trade charts | Equity+DD | Heatmaps | Full metrics                ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:  python orb_backtest_single.py
"""

# =============================================================================
# CONFIGURATION — Your winning combo
# =============================================================================
CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    "SPOT_DATA_PATH": "data/spot_parquet",
    "OPT_DATA_PATH":  "data/options_parquet",
    "SPOT_CANDLE_MINUTES": 5,

    # ── Date Range ────────────────────────────────────────────────────────────
    "DATE_FROM": "2025-01-01",
    "DATE_TO":   "2025-12-31",

    # ── Instrument ────────────────────────────────────────────────────────────
    "STRIKE_INTERVAL": 50,
    "LOT_SIZE":        75,
    "NUM_LOTS":        1,

    # ── ORB Strategy ──────────────────────────────────────────────────────────
    "ORB_START":       "09:15",
    "ORB_WINDOW_END":  "09:45",
    "DIRECTION":       "short_only",   # "both" | "long_only" | "short_only"
    "STRIKE_OFFSET":   0,              # 0=ATM, 1=OTM, -1=ITM
    "TRADE_MODE":      "buy",          # "buy" | "sell"

    # ── Exit Rules ────────────────────────────────────────────────────────────
    "USE_SPOT_SL":     True,
    "SPOT_SL_MODE":    "breakout_level",  # "opposite" | "breakout_level"
    "OPTION_SL_PTS":   None,           # Fixed SL in option pts (None=off)
    "OPTION_TGT_PTS":  100,            # Fixed target in option pts (None=off)
    "EOD_EXIT_TIME":   "14:00",

    # ── Expiry ────────────────────────────────────────────────────────────────
    "EXPIRY_CODE":     1,

    # ── Costs ─────────────────────────────────────────────────────────────────
    "BROKERAGE_PER_TRADE": 40,
    "SLIPPAGE_POINTS":     0.5,

    # ── Output ────────────────────────────────────────────────────────────────
    "OUTPUT_FOLDER":       "./orb_backtest_buy_plots",
    "TRADES_PER_PAGE":     6,          # Trade charts per PNG page
    "CHART_DPI":           150,
}


# =============================================================================
# IMPORTS
# =============================================================================
import os, sys, glob, warnings
from datetime import time, datetime, timedelta
from collections import defaultdict
import calendar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import engine as shared_engine

warnings.filterwarnings('ignore')

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try: stream.reconfigure(encoding='utf-8', errors='replace')
        except: pass

# ── Theme ─────────────────────────────────────────────────────────────────────
TH = {
    'bg': '#0D1117', 'panel': '#161B22', 'border': '#30363D', 'grid': '#21262D',
    'text': '#ECEFF4', 'muted': '#8B949E', 'green': '#2ECC71', 'red': '#E74C3C',
    'blue': '#3498DB', 'yellow': '#F1C40F', 'orange': '#E67E22', 'purple': '#9B59B6',
    'cyan': '#00BCD4',
}
PNL_CMAP = LinearSegmentedColormap.from_list('pnl', ['#E74C3C', '#1A1A2E', '#2ECC71'], N=256)
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def style_ax(ax):
    ax.set_facecolor(TH['panel'])
    ax.tick_params(colors=TH['muted'], labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(TH['border'])
    ax.grid(axis='y', color=TH['grid'], linewidth=0.5)


# =============================================================================
# HELPERS
# =============================================================================
def parse_time(t_str):
    h, m = map(int, t_str.split(':'))
    return time(h, m)

def get_strike(spot_price, offset, interval):
    atm = int(round(spot_price / interval) * interval)
    return atm + offset * interval

def apply_slippage(price, side, slippage):
    return price + slippage if side in ('entry_buy', 'close_short') else price - slippage

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
    return option_bar['close'], option_bar['time']


# =============================================================================
# DATA LOADING
# =============================================================================
def parse_ymd_date(value):
    return datetime.strptime(value, '%Y-%m-%d').date() if value else None

def month_end(ms):
    return (ms.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)

def dataset_file_matches_date_range(fp, date_from=None, date_to=None):
    sd, ed = parse_ymd_date(date_from), parse_ymd_date(date_to)
    stem = os.path.splitext(os.path.basename(fp))[0]
    try:
        if len(stem) == 10:
            fs = datetime.strptime(stem, '%Y-%m-%d').date(); fe = fs
        elif len(stem) == 7:
            fs = datetime.strptime(stem, '%Y-%m').date(); fe = month_end(fs)
        else: return True
    except ValueError: return True
    if sd and fe < sd: return False
    if ed and fs > ed: return False
    return True

def normalize_loaded_data(df, path):
    if 'datetime_ist' in df.columns:
        df['datetime_ist'] = pd.to_datetime(df['datetime_ist'], utc=True).dt.tz_convert('Asia/Kolkata')
    elif 'date' in df.columns:
        df['datetime_ist'] = pd.to_datetime(df['date'], dayfirst=True).dt.tz_localize('Asia/Kolkata')
        df = df.drop(columns=['date'])
    else:
        raise ValueError(f"No datetime column in {path}")
    df = df.sort_values('datetime_ist').reset_index(drop=True)
    df['date'] = df['datetime_ist'].dt.date
    df['time'] = df['datetime_ist'].dt.time
    return df

def load_dataset_folder(path, date_from=None, date_to=None, label='data'):
    pq = sorted(glob.glob(os.path.join(path, '**', '*.parquet'), recursive=True))
    cs = sorted(glob.glob(os.path.join(path, '**', '*.csv'), recursive=True))
    ft, files = ('parquet', pq) if pq else ('csv', cs) if cs else (None, [])
    if not ft: raise FileNotFoundError(f"No files under {path}")
    files = [f for f in files if dataset_file_matches_date_range(f, date_from, date_to)]
    if not files: raise FileNotFoundError(f"No {ft} files for {date_from}-{date_to}")
    dfs = []
    for f in files:
        try: dfs.append(pd.read_parquet(f) if ft == 'parquet' else pd.read_csv(f))
        except Exception as e: print(f"  Warn: {f} — {e}")
    df = pd.concat(dfs, ignore_index=True)
    df = normalize_loaded_data(df, path)
    if date_from: df = df[df['date'] >= datetime.strptime(date_from, '%Y-%m-%d').date()]
    if date_to:   df = df[df['date'] <= datetime.strptime(date_to, '%Y-%m-%d').date()]
    df = df.reset_index(drop=True)
    print(f"  {label:<8}: {len(df):>9,} rows | {df['date'].nunique():>4} days ({ft})")
    return df

def resample_spot(df, minutes):
    if minutes == 1: return df
    def _rs(g):
        r = g.set_index('datetime_ist').resample(f'{minutes}min', closed='left', label='left').agg(
            {'open':'first','high':'max','low':'min','close':'last'}).dropna(subset=['open'])
        return r
    out = df.groupby('date', group_keys=False).apply(_rs).reset_index()
    out['date'] = out['datetime_ist'].dt.date
    out['time'] = out['datetime_ist'].dt.time
    return out.sort_values('datetime_ist').reset_index(drop=True)

def load_data(cfg):
    spot_raw = load_dataset_folder(cfg['SPOT_DATA_PATH'], cfg['DATE_FROM'], cfg['DATE_TO'], 'Spot')
    opts = load_dataset_folder(cfg['OPT_DATA_PATH'],  cfg['DATE_FROM'], cfg['DATE_TO'], 'Options')
    opts = opts[opts['expiry_code'] == cfg['EXPIRY_CODE']].reset_index(drop=True)
    print(f"  Options after expiry filter (code={cfg['EXPIRY_CODE']}): {len(opts):,} rows")
    mins = cfg.get('SPOT_CANDLE_MINUTES', 1)
    if mins > 1:
        spot = resample_spot(spot_raw, mins)
        print(f"  Spot resampled to {mins}-min: {len(spot):,} bars | {spot['date'].nunique()} days")
    else:
        spot = spot_raw
    return spot, spot_raw, opts


# =============================================================================
# BACKTEST ENGINE — single config, full trade detail
# =============================================================================
def run_backtest(spot, spot_raw, opts, cfg):
    window_end  = parse_time(cfg['ORB_WINDOW_END'])
    eod_time    = parse_time(cfg['EOD_EXIT_TIME'])
    trade_mode  = cfg['TRADE_MODE']
    direction   = cfg['DIRECTION']
    strike_off  = cfg['STRIKE_OFFSET']
    interval    = cfg['STRIKE_INTERVAL']
    lot_size    = cfg['LOT_SIZE']
    num_lots    = cfg['NUM_LOTS']
    slippage    = cfg['SLIPPAGE_POINTS']
    use_spot_sl = cfg['USE_SPOT_SL']
    spot_sl_mode = cfg['SPOT_SL_MODE']
    spot_candle_minutes = cfg.get('SPOT_CANDLE_MINUTES', 1)
    sl_pts      = cfg['OPTION_SL_PTS']
    tgt_pts     = cfg['OPTION_TGT_PTS']
    brk_pts     = brokerage_in_pts(lot_size, num_lots, cfg['BROKERAGE_PER_TRADE'])

    trades = []
    for trade_date, day_spot in spot.groupby('date'):
        day_spot = day_spot.sort_values('time').reset_index(drop=True)
        day_opts = opts[opts['date'] == trade_date].copy()
        if day_opts.empty:
            continue

        or_bars   = day_spot[day_spot['time'] < window_end]
        # Spot bars are left-labeled after resampling, so a 09:45 bar covers
        # 09:45-09:50 and closes at 09:50. That first post-ORB bar must be
        # eligible to trigger an entry.
        post_bars = day_spot[(day_spot['time'] >= window_end) & (day_spot['time'] <= eod_time)]
        if len(or_bars) < 1 or len(post_bars) < 1:
            continue

        or_high = or_bars['high'].max()
        or_low  = or_bars['low'].min()

        trade_taken = False
        for _, bar in post_bars.iterrows():
            if trade_taken: break
            sig_dir = None
            if bar['close'] > or_high and direction in ('both', 'long_only'):
                sig_dir = 'Long'
            elif bar['close'] < or_low and direction in ('both', 'short_only'):
                sig_dir = 'Short'
            if sig_dir is None: continue
            trade_taken = True

            signal_time = bar['time']
            # Entry after the 5-min candle closes (e.g. 10:00 candle → enter at 10:05)
            signal_close_time = shift_time_by_minutes(signal_time, spot_candle_minutes)
            spot_price  = bar['close']
            opt_type = ('CE' if sig_dir == 'Long' else 'PE') if trade_mode == 'buy' else \
                       ('PE' if sig_dir == 'Long' else 'CE')

            if spot_sl_mode == 'breakout_level':
                spot_sl = or_high if sig_dir == 'Long' else or_low
            else:
                spot_sl = or_low if sig_dir == 'Long' else or_high

            eff_offset = strike_off if opt_type == 'CE' else -strike_off
            strike = get_strike(spot_price, eff_offset, interval)

            day_opt_f = day_opts[
                (day_opts['actual_strike'] == strike) &
                (day_opts['option_type']   == opt_type)
            ].sort_values('time')
            eb = day_opt_f[day_opt_f['time'] >= signal_close_time]
            if eb.empty: continue
            raw_entry  = eb.iloc[0]['open']
            entry_time = eb.iloc[0]['time']
            if raw_entry <= 0: continue

            entry_price = apply_slippage(raw_entry,
                'entry_buy' if trade_mode == 'buy' else 'entry_sell', slippage)

            # SL / Target levels
            if trade_mode == 'buy':
                sl_price  = (entry_price - sl_pts)  if sl_pts is not None else None
                tgt_price = (entry_price + tgt_pts) if tgt_pts is not None else None
            else:
                sl_price  = (entry_price + sl_pts)  if sl_pts is not None else None
                tgt_price = (entry_price - tgt_pts) if tgt_pts is not None else None

            # ── Walk forward ─────────────────────────────────────────────────
            post_opt = day_opt_f[(day_opt_f['time'] > entry_time) &
                                 (day_opt_f['time'] <= eod_time)].reset_index(drop=True)
            # Use 1-min spot data for exit scanning (touch detection)
            day_spot_raw = spot_raw[spot_raw['date'] == trade_date].sort_values('time')
            post_spot_1m = day_spot_raw[(day_spot_raw['time'] > entry_time) &
                                        (day_spot_raw['time'] <= eod_time)]

            exit_price  = None
            exit_reason = 'EOD'
            exit_time   = None

            spot_idx = 0
            post_spot_list = post_spot_1m.to_dict('records')
            post_opt_list = post_opt.to_dict('records')

            for obar_idx, obar in enumerate(post_opt_list):
                curr_t    = obar['time']
                opt_close = obar['close']
                opt_high  = obar['high']
                opt_low_b = obar['low']
                spot_sl_hit = False
                while use_spot_sl and spot_idx < len(post_spot_list) and \
                        shift_time_by_minutes(post_spot_list[spot_idx]['time'], 1) <= curr_t:
                    spot_bar = post_spot_list[spot_idx]
                    if sig_dir == 'Long' and spot_bar['low'] <= spot_sl: spot_sl_hit = True
                    if sig_dir == 'Short' and spot_bar['high'] >= spot_sl: spot_sl_hit = True
                    spot_idx += 1
                while not use_spot_sl and spot_idx < len(post_spot_list) and \
                        shift_time_by_minutes(post_spot_list[spot_idx]['time'], 1) <= curr_t:
                    spot_idx += 1

                if spot_sl_hit:
                    exit_price, exit_time = resolve_spot_sl_exit(obar)
                    exit_reason = 'SPOT_SL'; break

                if trade_mode == 'buy':
                    opt_sl_hit  = sl_price  is not None and opt_low_b <= sl_price
                    opt_tgt_hit = tgt_price is not None and opt_high  >= tgt_price
                else:
                    opt_sl_hit  = sl_price  is not None and opt_high >= sl_price
                    opt_tgt_hit = tgt_price is not None and opt_low_b <= tgt_price

                if opt_tgt_hit:
                    exit_price = tgt_price; exit_reason = 'TARGET'; exit_time = curr_t; break
                elif opt_sl_hit:
                    exit_price = sl_price;  exit_reason = 'OPT_SL'; exit_time = curr_t; break
                else:
                    exit_price = opt_close
                    exit_time  = curr_t

            if exit_price is None:
                exit_price = entry_price; exit_reason = 'NO_DATA'

            # P&L
            if trade_mode == 'buy':
                raw_exit = apply_slippage(exit_price, 'close_long', slippage)
                raw_pnl  = raw_exit - entry_price
            else:
                raw_exit = apply_slippage(exit_price, 'close_short', slippage)
                raw_pnl  = entry_price - raw_exit

            pnl_pts = raw_pnl - brk_pts
            pnl_inr = pnl_pts * lot_size * num_lots

            trades.append({
                'date':           trade_date,
                'day_of_week':    trade_date.strftime('%A'),
                'direction':      sig_dir,
                'opt_type':       opt_type,
                'strike':         strike,
                'atm':            get_strike(spot_price, 0, interval),
                'signal_time':    signal_time,
                'trigger_time':   signal_time,
                'entry_time':     entry_time,
                'exit_time':      exit_time,
                'or_high':        round(or_high, 2),
                'or_low':         round(or_low, 2),
                'spot_sl':        round(spot_sl, 2),
                'spot_at_signal': round(spot_price, 2),
                'entry_price':    round(entry_price, 2),
                'exit_price':     round(exit_price, 2),
                'target_price':   round(tgt_price, 2) if tgt_price is not None else None,
                'sl_level':       round(sl_price, 2)  if sl_price is not None else None,
                'exit_reason':    exit_reason,
                'pnl_pts':        round(pnl_pts, 2),
                'pnl_inr':        round(pnl_inr, 2),
            })

    return pd.DataFrame(trades)


# =============================================================================
# STATISTICS
# =============================================================================
def compute_stats(df, cfg):
    if df.empty:
        return {}
    pnl = df['pnl_pts']
    cum = pnl.cumsum()
    wins = df[pnl > 0]; losses = df[pnl <= 0]
    wr = len(wins) / len(df) * 100
    avg_w = wins['pnl_pts'].mean() if len(wins) > 0 else 0
    avg_l = losses['pnl_pts'].mean() if len(losses) > 0 else 0
    rr = abs(avg_w / avg_l) if avg_l != 0 else 0
    gw = wins['pnl_pts'].sum() if len(wins) > 0 else 0
    gl = abs(losses['pnl_pts'].sum()) if len(losses) > 0 else 0
    pf = gw / gl if gl > 0 else (99.0 if gw > 0 else 0)
    run_max = cum.cummax()
    dd = run_max - cum
    mdd = dd.max()

    # Streaks
    flags = (pnl > 0).tolist()
    mws = mls = cw = cl = 0
    for v in flags:
        if v: cw += 1; cl = 0
        else: cl += 1; cw = 0
        mws = max(mws, cw); mls = max(mls, cl)

    long_df = df[df['direction'] == 'Long']; short_df = df[df['direction'] == 'Short']
    lwr = len(long_df[long_df['pnl_pts'] > 0]) / len(long_df) * 100 if len(long_df) > 0 else 0
    swr = len(short_df[short_df['pnl_pts'] > 0]) / len(short_df) * 100 if len(short_df) > 0 else 0

    lot = cfg['LOT_SIZE'] * cfg['NUM_LOTS']
    return {
        'total_trades':    len(df),
        'win_rate':        round(wr, 1),
        'total_pnl_pts':   round(cum.iloc[-1], 2),
        'total_pnl_inr':   round(df['pnl_inr'].sum(), 0),
        'avg_pnl_pts':     round(pnl.mean(), 2),
        'avg_win_pts':     round(avg_w, 2),
        'avg_loss_pts':    round(avg_l, 2),
        'rr_ratio':        round(rr, 2),
        'profit_factor':   round(pf, 2),
        'max_dd_pts':      round(mdd, 2),
        'max_dd_inr':      round(mdd * lot, 0),
        'max_win_streak':  mws,
        'max_loss_streak': mls,
        'long_trades':     len(long_df),
        'short_trades':    len(short_df),
        'long_wr':         round(lwr, 1),
        'short_wr':        round(swr, 1),
        'exit_counts':     df['exit_reason'].value_counts().to_dict(),
        'best_trade_pts':  round(pnl.max(), 2),
        'worst_trade_pts': round(pnl.min(), 2),
        'median_pnl_pts':  round(pnl.median(), 2),
        'std_pnl_pts':     round(pnl.std(ddof=0) if len(df) > 1 else 0.0, 2),
        'avg_hold_mins':   0,  # filled below
    }


# =============================================================================
# CHART 1 — Per-trade charts (spot top + option bottom), paginated
# =============================================================================
def draw_candles(ax, df, alpha=1.0):
    for i, (_, row) in enumerate(df.iterrows()):
        col = TH['green'] if row['close'] >= row['open'] else TH['red']
        ax.plot([i, i], [row['low'], row['high']], color=col, lw=0.8, zorder=2, alpha=alpha)
        body = max(abs(row['close'] - row['open']), 0.15)
        ax.add_patch(plt.Rectangle(
            (i - 0.35, min(row['open'], row['close'])), 0.7, body,
            color=col, zorder=3, alpha=alpha))

def set_time_xticks(ax, times, step=6):
    times = times.reset_index(drop=True)
    pos = list(range(0, len(times), step))
    ax.set_xticks(pos)
    ax.set_xticklabels([times.iloc[i].strftime('%H:%M') for i in pos],
                       fontsize=6, color=TH['muted'], rotation=45)
    ax.set_xlim(-1, len(times))


def plot_trade_pages(trades_df, spot, spot_raw, opts, cfg, out_dir):
    """Generate paginated trade chart PNGs. Returns list of paths."""
    if trades_df.empty: return []

    window_end = parse_time(cfg['ORB_WINDOW_END'])
    eod_time   = parse_time(cfg['EOD_EXIT_TIME'])
    mode       = cfg['TRADE_MODE'].upper()
    per_page   = cfg['TRADES_PER_PAGE']
    paths      = []

    n = len(trades_df)
    n_pages = (n + per_page - 1) // per_page

    for page in range(n_pages):
        start = page * per_page
        end   = min(start + per_page, n)
        page_trades = trades_df.iloc[start:end]
        n_t = len(page_trades)

        fig = plt.figure(figsize=(22, n_t * 8), facecolor=TH['bg'])
        fig.suptitle(
            f"ORB {mode} Trade Charts  |  Page {page+1}/{n_pages}  |  "
            f"Trades {start+1}-{end} of {n}",
            fontsize=13, color=TH['text'], fontweight='bold', y=0.998)
        outer_gs = gridspec.GridSpec(n_t, 1, figure=fig, hspace=0.55,
                                     top=0.993, bottom=0.01, left=0.05, right=0.97)

        for idx, trade in enumerate(page_trades.itertuples()):
            date    = trade.date
            sig_t   = trade.trigger_time
            entry_p = trade.entry_price
            exit_p  = trade.exit_price
            pnl     = trade.pnl_pts
            dirn    = trade.direction
            strike  = trade.strike
            ot      = trade.opt_type
            ex_reas = trade.exit_reason
            spot_sl = trade.spot_sl
            tgt_p   = getattr(trade, 'target_price', None)

            day_spot = spot[spot['date'] == date].sort_values('time').reset_index(drop=True)
            day_spot_1m = spot_raw[spot_raw['date'] == date].sort_values('time').reset_index(drop=True)
            day_opt  = opts[(opts['date'] == date) &
                            (opts['actual_strike'] == strike) &
                            (opts['option_type'] == ot)].sort_values('time').reset_index(drop=True)
            if day_spot.empty or day_spot_1m.empty: continue

            or_bars  = day_spot[day_spot['time'] < window_end]
            or_high  = or_bars['high'].max() if len(or_bars) > 0 else 0
            or_low   = or_bars['low'].min()  if len(or_bars) > 0 else 0
            or_end_i = or_bars.index[-1]     if len(or_bars) > 0 else 0

            def map_t(t):
                c = day_spot[day_spot['time'] <= t]
                return c.index[-1] if len(c) > 0 else 0

            sig_xi = map_t(sig_t)

            # Exit x-index
            if ex_reas == 'SPOT_SL':
                exit_xi = map_t(trade.exit_time) if pd.notna(trade.exit_time) else len(day_spot) - 1
            elif ex_reas == 'TARGET' and tgt_p:
                tob = day_opt[(day_opt['time'] > sig_t) & (day_opt['high'] >= tgt_p)]
                exit_xi = map_t(tob.iloc[0]['time']) if len(tob) > 0 else len(day_spot) - 1
            else:
                eodb = day_spot[day_spot['time'] >= eod_time]
                exit_xi = eodb.index[0] if len(eodb) > 0 else len(day_spot) - 1

            pnl_color = TH['green'] if pnl >= 0 else TH['red']

            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[idx],
                hspace=0.08, height_ratios=[1.6, 1])

            # ── Spot Panel ───────────────────────────────────────────────────
            ax_s = fig.add_subplot(inner[0])
            draw_candles(ax_s, day_spot)
            ax_s.axvspan(0, or_end_i, alpha=0.08, color=TH['blue'])
            ax_s.axhline(or_high, color=TH['blue'],   lw=1, ls='--', alpha=0.7)
            ax_s.axhline(or_low,  color=TH['orange'], lw=1, ls='--', alpha=0.7)
            ax_s.axhline(spot_sl, color=TH['red'],    lw=0.8, ls=':', alpha=0.5)

            e_marker = '^' if dirn == 'Long' else 'v'
            ax_s.scatter(sig_xi, day_spot.iloc[sig_xi]['close'],
                         marker=e_marker, color=TH['yellow'], s=180, zorder=7)
            ax_s.scatter(exit_xi, day_spot.iloc[exit_xi]['close'],
                         marker='s', color=pnl_color, s=140, zorder=7)
            ax_s.axvspan(sig_xi, exit_xi, alpha=0.06, color=pnl_color)

            ax_s.text(or_end_i + 0.3, or_high, f'OR H:{or_high:.0f}',
                      color=TH['blue'], fontsize=6.5, va='bottom')
            ax_s.text(or_end_i + 0.3, or_low, f'OR L:{or_low:.0f}',
                      color=TH['orange'], fontsize=6.5, va='top')

            set_time_xticks(ax_s, day_spot['time'])
            style_ax(ax_s)
            pnl_sign = '+' if pnl >= 0 else ''
            ax_s.set_title(
                f"#{start+idx+1}  {date}  {trade.day_of_week[:3]}  |  "
                f"{dirn} -> {mode} {strike}{ot}  |  "
                f"Entry:{entry_p:.1f}  Exit:{exit_p:.1f}  |  "
                f"P&L: {pnl_sign}{pnl:.1f}pts  [{ex_reas}]",
                color=pnl_color, fontsize=9, fontweight='bold', pad=5)

            # ── Option Panel ─────────────────────────────────────────────────
            ax_o = fig.add_subplot(inner[1])
            if len(day_opt) > 0:
                for _, orow in day_opt.iterrows():
                    xi = map_t(orow['time'])
                    col = TH['green'] if orow['close'] >= orow['open'] else TH['red']
                    ax_o.plot([xi, xi], [orow['low'], orow['high']],
                              color=col, lw=0.6, zorder=2, alpha=0.8)
                    bh = max(abs(orow['close'] - orow['open']), 0.1)
                    ax_o.add_patch(plt.Rectangle(
                        (xi - 0.28, min(orow['open'], orow['close'])),
                        0.56, bh, color=col, zorder=3, alpha=0.85))

                ax_o.axhline(entry_p, color=TH['yellow'], lw=0.9, ls='--', alpha=0.75,
                             label=f"Entry {entry_p:.0f}")
                if tgt_p:
                    ax_o.axhline(tgt_p, color=TH['green'], lw=0.9, ls=':', alpha=0.85,
                                 label=f"Target {tgt_p:.0f}")
                if trade.sl_level and not pd.isna(trade.sl_level):
                    ax_o.axhline(trade.sl_level, color=TH['red'], lw=0.9, ls=':', alpha=0.7,
                                 label=f"SL {trade.sl_level:.0f}")

                post_opt = day_opt[day_opt['time'] >= sig_t]
                if len(post_opt) > 0:
                    pxi = [map_t(t) for t in post_opt['time']]
                    pc  = post_opt['close'].values
                    if cfg['TRADE_MODE'] == 'buy':
                        ax_o.fill_between(pxi, entry_p, pc,
                                          where=pc > entry_p, color=TH['green'], alpha=0.12)
                        ax_o.fill_between(pxi, entry_p, pc,
                                          where=pc <= entry_p, color=TH['red'], alpha=0.12)
                    else:
                        ax_o.fill_between(pxi, entry_p, pc,
                                          where=pc < entry_p, color=TH['green'], alpha=0.12)
                        ax_o.fill_between(pxi, entry_p, pc,
                                          where=pc >= entry_p, color=TH['red'], alpha=0.12)

                ax_o.scatter(sig_xi, entry_p, marker=e_marker, color=TH['yellow'], s=100, zorder=6)
                ax_o.scatter(exit_xi, exit_p, marker='s', color=pnl_color, s=80, zorder=6)
                ax_o.legend(fontsize=6, facecolor='#1C2128', edgecolor=TH['border'],
                            labelcolor=TH['text'], loc='upper right')

            set_time_xticks(ax_o, day_spot['time'])
            style_ax(ax_o)
            ax_o.set_ylabel(f'{ot} (pts)', color=TH['muted'], fontsize=7)
            ax_o.set_xlim(-1, len(day_spot))

        fname = os.path.join(out_dir, f"trades_page_{page+1:03d}.png")
        plt.savefig(fname, dpi=cfg['CHART_DPI'], bbox_inches='tight', facecolor=TH['bg'])
        plt.close(fig)
        paths.append(fname)
        print(f"    Page {page+1}/{n_pages} -> {fname}")

    return paths


# =============================================================================
# CHART 2 — Equity Curve + Drawdown
# =============================================================================
def plot_equity_drawdown(df, stats, cfg, out_path):
    cum = df['pnl_pts'].cumsum()
    cum_inr = df['pnl_inr'].cumsum()
    lot = cfg['LOT_SIZE'] * cfg['NUM_LOTS']
    run_max = cum.cummax()
    dd = run_max - cum

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), facecolor=TH['bg'],
                                     gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.15})

    mode = cfg['TRADE_MODE'].upper()
    fig.suptitle(
        f"ORB {mode} {cfg['DIRECTION']}  |  OR {cfg['ORB_START']}-{cfg['ORB_WINDOW_END']}  |  "
        f"SL:{cfg['OPTION_SL_PTS'] or 'Off'}  Tgt:{cfg['OPTION_TGT_PTS'] or 'Off'}  "
        f"EOD:{cfg['EOD_EXIT_TIME']}  SpotSL:{cfg['SPOT_SL_MODE']}",
        fontsize=12, color=TH['text'], fontweight='bold')

    # Equity
    style_ax(ax1)
    x = range(len(cum))
    cum_vals = cum.values
    ax1.fill_between(x, 0, cum_vals, where=cum_vals >= 0, color=TH['green'], alpha=0.15)
    ax1.fill_between(x, 0, cum_vals, where=cum_vals < 0,  color=TH['red'],   alpha=0.15)
    ax1.plot(cum_vals, color=TH['blue'], lw=1.8)
    ax1.axhline(0, color='#555', lw=0.8, ls='--')
    ax1.set_ylabel('Cumulative P&L (pts)', color=TH['muted'], fontsize=9)
    ax1.set_title(
        f"{stats['total_trades']} trades  |  WR: {stats['win_rate']}%  |  "
        f"PF: {stats['profit_factor']}  |  R:R: {stats['rr_ratio']}  |  "
        f"P&L: {stats['total_pnl_pts']:.0f}pts / Rs.{stats['total_pnl_inr']:,.0f}",
        color=TH['text'], fontsize=10, pad=6)

    # Drawdown
    style_ax(ax2)
    ax2.fill_between(x, 0, -dd.values, color=TH['red'], alpha=0.4)
    ax2.plot(-dd.values, color=TH['red'], lw=1)
    ax2.set_ylabel('Drawdown (pts)', color=TH['muted'], fontsize=9)
    ax2.set_xlabel('Trade #', color=TH['muted'], fontsize=9)
    ax2.set_title(f"Max Drawdown: {stats['max_dd_pts']:.1f} pts / Rs.{stats['max_dd_inr']:,.0f}",
                  color=TH['red'], fontsize=10, pad=4)

    plt.savefig(out_path, dpi=cfg['CHART_DPI'], bbox_inches='tight', facecolor=TH['bg'])
    plt.close(fig)


# =============================================================================
# CHART 3 — Monthly P&L Heatmap (Year × Month)
# =============================================================================
def plot_monthly_heatmap(df, cfg, out_path):
    df = df.copy()
    df['year']  = df['date'].apply(lambda d: d.year)
    df['month'] = df['date'].apply(lambda d: d.month)
    monthly = df.groupby(['year', 'month'])['pnl_pts'].sum().reset_index()
    years = sorted(df['year'].unique())

    grid = np.full((len(years), 12), np.nan)
    for _, row in monthly.iterrows():
        yi = years.index(row['year'])
        grid[yi, int(row['month']) - 1] = row['pnl_pts']

    vabs = np.nanmax(np.abs(grid[~np.isnan(grid)])) if np.any(~np.isnan(grid)) else 1

    fig, ax = plt.subplots(figsize=(16, max(3, len(years) * 1.2)), facecolor=TH['bg'])
    im = ax.imshow(grid, cmap=PNL_CMAP, aspect='auto', vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_NAMES, color=TH['text'], fontsize=9)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, color=TH['text'], fontsize=9)
    ax.set_facecolor(TH['bg'])

    for yi in range(len(years)):
        for mi in range(12):
            val = grid[yi, mi]
            if np.isnan(val): continue
            color = '#000' if abs(val) > vabs * 0.55 else TH['text']
            ax.text(mi, yi, f'{val:.0f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    # Row totals
    for yi, yr in enumerate(years):
        total = np.nansum(grid[yi])
        col = TH['green'] if total >= 0 else TH['red']
        ax.text(12.3, yi, f'{total:.0f}', ha='left', va='center',
                fontsize=9, color=col, fontweight='bold')
    ax.text(12.3, -0.7, 'Year Total', ha='left', va='center',
            fontsize=8, color=TH['muted'], fontweight='bold')

    fig.colorbar(im, ax=ax, label='P&L (pts)', shrink=0.8, pad=0.12)
    ax.set_title('Monthly P&L Heatmap (pts)', color=TH['text'], fontsize=13,
                 fontweight='bold', pad=10)
    plt.savefig(out_path, dpi=cfg['CHART_DPI'], bbox_inches='tight', facecolor=TH['bg'])
    plt.close(fig)


# =============================================================================
# CHART 4 — Day-of-Week Breakdown
# =============================================================================
def plot_dow_breakdown(df, cfg, out_path):
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df = df.copy()
    df['dow'] = df['date'].apply(lambda d: d.strftime('%A'))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=TH['bg'])
    fig.suptitle('Day-of-Week Breakdown', color=TH['text'], fontsize=13, fontweight='bold')

    # Total PnL per day
    ax = axes[0]; style_ax(ax)
    grp = df.groupby('dow')['pnl_pts'].sum().reindex(dow_order, fill_value=0)
    colors = [TH['green'] if v >= 0 else TH['red'] for v in grp.values]
    bars = ax.bar(range(len(grp)), grp.values, color=colors, edgecolor=TH['bg'])
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([d[:3] for d in dow_order], color=TH['muted'], fontsize=9)
    ax.set_title('Total P&L (pts)', color=TH['text'], fontsize=10, pad=4)
    ax.axhline(0, color='#555', lw=0.8)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(val)*0.03,
                f'{val:.0f}', ha='center', fontsize=7.5, color=TH['text'])

    # Win Rate per day
    ax = axes[1]; style_ax(ax)
    wr = df.groupby('dow').apply(lambda g: len(g[g['pnl_pts'] > 0]) / len(g) * 100
                                  if len(g) > 0 else 0).reindex(dow_order, fill_value=0)
    colors = [TH['green'] if v >= 50 else TH['orange'] for v in wr.values]
    bars = ax.bar(range(len(wr)), wr.values, color=colors, edgecolor=TH['bg'])
    ax.set_xticks(range(len(wr)))
    ax.set_xticklabels([d[:3] for d in dow_order], color=TH['muted'], fontsize=9)
    ax.set_title('Win Rate %', color=TH['text'], fontsize=10, pad=4)
    ax.axhline(50, color=TH['red'], lw=0.8, ls='--', alpha=0.5)
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, wr.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', fontsize=7.5, color=TH['text'])

    # Trade count per day
    ax = axes[2]; style_ax(ax)
    cnt = df.groupby('dow').size().reindex(dow_order, fill_value=0)
    ax.bar(range(len(cnt)), cnt.values, color=TH['blue'], edgecolor=TH['bg'])
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels([d[:3] for d in dow_order], color=TH['muted'], fontsize=9)
    ax.set_title('Trade Count', color=TH['text'], fontsize=10, pad=4)
    for i, val in enumerate(cnt.values):
        ax.text(i, val + 0.5, str(val), ha='center', fontsize=7.5, color=TH['text'])

    plt.savefig(out_path, dpi=cfg['CHART_DPI'], bbox_inches='tight', facecolor=TH['bg'])
    plt.close(fig)


# =============================================================================
# CHART 5 — P&L Distribution + Exit Breakdown
# =============================================================================
def plot_distribution_and_exits(df, cfg, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), facecolor=TH['bg'])
    fig.suptitle('P&L Distribution & Exit Breakdown', color=TH['text'],
                 fontsize=13, fontweight='bold')

    # Histogram
    style_ax(ax1)
    pnls = df['pnl_pts'].values
    bins = np.linspace(pnls.min() - 5, pnls.max() + 5, 40)
    n, b, patches = ax1.hist(pnls, bins=bins, edgecolor=TH['bg'], linewidth=0.5)
    for patch, left in zip(patches, b[:-1]):
        patch.set_facecolor(TH['green'] if left >= 0 else TH['red'])
        patch.set_alpha(0.8)
    ax1.axvline(0, color=TH['yellow'], lw=1, ls='--', alpha=0.7)
    ax1.axvline(pnls.mean(), color=TH['cyan'], lw=1, ls=':', alpha=0.8,
                label=f'Mean: {pnls.mean():.1f}')
    ax1.axvline(np.median(pnls), color=TH['purple'], lw=1, ls=':', alpha=0.8,
                label=f'Median: {np.median(pnls):.1f}')
    ax1.set_xlabel('P&L (pts)', color=TH['muted'], fontsize=9)
    ax1.set_ylabel('Frequency', color=TH['muted'], fontsize=9)
    ax1.set_title('P&L Distribution', color=TH['text'], fontsize=10, pad=4)
    ax1.legend(fontsize=8, facecolor='#1C2128', edgecolor=TH['border'], labelcolor=TH['text'])

    # Exit breakdown donut
    ax2.set_facecolor(TH['bg'])
    exit_counts = df['exit_reason'].value_counts()
    exit_colors_map = {
        'EOD': TH['blue'], 'SPOT_SL': TH['red'], 'TARGET': TH['green'],
        'OPT_SL': TH['orange'], 'NO_DATA': TH['muted'],
    }
    colors = [exit_colors_map.get(e, TH['muted']) for e in exit_counts.index]
    wedges, texts, autotexts = ax2.pie(
        exit_counts.values, labels=exit_counts.index, autopct='%1.0f%%',
        colors=colors, pctdistance=0.78, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor=TH['bg'], linewidth=2))
    for t in texts: t.set_color(TH['text']); t.set_fontsize(9)
    for t in autotexts: t.set_color('#FFF'); t.set_fontsize(8); t.set_fontweight('bold')
    ax2.set_title('Exit Type Breakdown', color=TH['text'], fontsize=10, pad=4)

    plt.savefig(out_path, dpi=cfg['CHART_DPI'], bbox_inches='tight', facecolor=TH['bg'])
    plt.close(fig)


# =============================================================================
# CHART 6 — Summary Stats Image
# =============================================================================
def plot_summary_table(stats, cfg, out_path):
    mode = cfg['TRADE_MODE'].upper()
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=TH['bg'])
    ax.set_facecolor(TH['bg']); ax.axis('off')
    fig.suptitle(
        f"ORB {mode} Backtest Summary  |  {cfg['DIRECTION']}  |  "
        f"OR {cfg['ORB_START']}-{cfg['ORB_WINDOW_END']}",
        color=TH['text'], fontsize=14, fontweight='bold', y=0.97)

    rows = [
        ['Total Trades',     str(stats['total_trades'])],
        ['Win Rate',         f"{stats['win_rate']}%"],
        ['Total P&L (pts)',  f"{stats['total_pnl_pts']:.1f}"],
        ['Total P&L (INR)',  f"Rs.{stats['total_pnl_inr']:,.0f}"],
        ['Avg P&L (pts)',    f"{stats['avg_pnl_pts']:.2f}"],
        ['Avg Win (pts)',    f"{stats['avg_win_pts']:.2f}"],
        ['Avg Loss (pts)',   f"{stats['avg_loss_pts']:.2f}"],
        ['R:R Ratio',        f"{stats['rr_ratio']:.2f}"],
        ['Profit Factor',    f"{stats['profit_factor']:.2f}"],
        ['Max Drawdown',     f"{stats['max_dd_pts']:.1f} pts / Rs.{stats['max_dd_inr']:,.0f}"],
        ['Best Trade',       f"{stats['best_trade_pts']:.1f} pts"],
        ['Worst Trade',      f"{stats['worst_trade_pts']:.1f} pts"],
        ['Median P&L',       f"{stats['median_pnl_pts']:.1f} pts"],
        ['Std Dev',          f"{stats['std_pnl_pts']:.1f} pts"],
        ['Max Win Streak',   str(stats['max_win_streak'])],
        ['Max Loss Streak',  str(stats['max_loss_streak'])],
        ['Long Trades',      f"{stats['long_trades']}  (WR: {stats['long_wr']}%)"],
        ['Short Trades',     f"{stats['short_trades']}  (WR: {stats['short_wr']}%)"],
    ]
    for reason, count in stats['exit_counts'].items():
        rows.append([f'Exit: {reason}', str(count)])

    # Config section
    rows.append(['', ''])
    rows.append(['--- CONFIG ---', ''])
    rows.append(['ORB Window',     f"{cfg['ORB_START']} - {cfg['ORB_WINDOW_END']}"])
    rows.append(['Candle Size',    f"{cfg['SPOT_CANDLE_MINUTES']} min"])
    rows.append(['Direction',      cfg['DIRECTION']])
    rows.append(['Trade Mode',     cfg['TRADE_MODE'].upper()])
    rows.append(['Strike',         f"ATM{cfg['STRIKE_OFFSET']:+d}"])
    rows.append(['Option SL',      f"{cfg['OPTION_SL_PTS']} pts" if cfg['OPTION_SL_PTS'] else 'Off'])
    rows.append(['Option Target',  f"{cfg['OPTION_TGT_PTS']} pts" if cfg['OPTION_TGT_PTS'] else 'Off'])
    rows.append(['EOD Exit',       cfg['EOD_EXIT_TIME']])
    rows.append(['Spot SL Mode',   cfg['SPOT_SL_MODE']])
    rows.append(['Costs',          f"Rs.{cfg['BROKERAGE_PER_TRADE']}/trade + "
                                   f"{cfg['SLIPPAGE_POINTS']}pt slip"])
    rows.append(['Position',       f"{cfg['NUM_LOTS']} x {cfg['LOT_SIZE']}"])
    rows.append(['Date Range',     f"{cfg['DATE_FROM']} to {cfg['DATE_TO']}"])

    table = ax.table(cellText=rows, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center', colWidths=[0.45, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(TH['border'])
        if r == 0:
            cell.set_facecolor('#1F6FEB')
            cell.set_text_props(color=TH['text'], fontweight='bold')
        else:
            txt = rows[r-1][0] if c == 0 else rows[r-1][1]
            if '--- CONFIG ---' in rows[r-1][0]:
                cell.set_facecolor('#1F3A5F')
                cell.set_text_props(color=TH['yellow'], fontweight='bold')
            elif rows[r-1][0] == 'Total P&L (pts)':
                pval = stats['total_pnl_pts']
                cell.set_facecolor('#1A2B1A' if pval >= 0 else '#2B1A1A')
                cell.set_text_props(color=TH['green'] if pval >= 0 else TH['red'],
                                    fontweight='bold')
            else:
                cell.set_facecolor(TH['panel'])
                cell.set_text_props(color=TH['text'])

    plt.savefig(out_path, dpi=cfg['CHART_DPI'], bbox_inches='tight', facecolor=TH['bg'])
    plt.close(fig)


# =============================================================================
# CHART 7 — Yearly Comparison Bars
# =============================================================================
def plot_yearly_bars(df, cfg, out_path):
    df = df.copy()
    df['year'] = df['date'].apply(lambda d: d.year)
    yearly = df.groupby('year').agg(
        pnl=('pnl_pts', 'sum'),
        trades=('pnl_pts', 'count'),
        wr=('pnl_pts', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0)
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=TH['bg'])
    fig.suptitle('Yearly Performance', color=TH['text'], fontsize=13, fontweight='bold')

    # PnL bars
    style_ax(ax1)
    colors = [TH['green'] if v >= 0 else TH['red'] for v in yearly['pnl']]
    bars = ax1.bar(yearly['year'].astype(str), yearly['pnl'], color=colors, edgecolor=TH['bg'])
    ax1.axhline(0, color='#555', lw=0.8)
    ax1.set_title('Total P&L (pts) per Year', color=TH['text'], fontsize=10, pad=4)
    for bar, val in zip(bars, yearly['pnl']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(val)*0.03,
                f'{val:.0f}', ha='center', fontsize=8, color=TH['text'])

    # WR bars
    style_ax(ax2)
    colors = [TH['green'] if v >= 50 else TH['orange'] for v in yearly['wr']]
    bars = ax2.bar(yearly['year'].astype(str), yearly['wr'], color=colors, edgecolor=TH['bg'])
    ax2.axhline(50, color=TH['red'], lw=0.8, ls='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_title('Win Rate % per Year', color=TH['text'], fontsize=10, pad=4)
    for bar, val, n in zip(bars, yearly['wr'], yearly['trades']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.0f}% ({n})', ha='center', fontsize=7.5, color=TH['text'])

    plt.savefig(out_path, dpi=cfg['CHART_DPI'], bbox_inches='tight', facecolor=TH['bg'])
    plt.close(fig)


# Reuse the shared engine so the standalone plotting script stays aligned with
# the API-backed backtester for buy/sell flows and config validation.
load_data = shared_engine.load_data
run_backtest = shared_engine.run_backtest
compute_stats = shared_engine.compute_stats


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg = CONFIG
    out = cfg['OUTPUT_FOLDER']
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'trade_charts'), exist_ok=True)

    mode = cfg['TRADE_MODE'].upper()
    print("\n" + "=" * 70)
    print("  ORB Options — Single Config Backtest")
    print("=" * 70)
    print(f"  Mode         : {mode}")
    print(f"  Direction    : {cfg['DIRECTION']}")
    print(f"  ORB Window   : {cfg['ORB_START']} - {cfg['ORB_WINDOW_END']}  "
          f"({cfg['SPOT_CANDLE_MINUTES']}min candles)")
    print(f"  Strike       : ATM{cfg['STRIKE_OFFSET']:+d}")
    print(f"  Option SL    : {cfg['OPTION_SL_PTS'] or 'Off'}")
    print(f"  Option Target: {cfg['OPTION_TGT_PTS'] or 'Off'}")
    print(f"  EOD Exit     : {cfg['EOD_EXIT_TIME']}")
    print(f"  Spot SL      : {cfg['SPOT_SL_MODE']}")
    print(f"  Costs        : Rs.{cfg['BROKERAGE_PER_TRADE']}/trade + {cfg['SLIPPAGE_POINTS']}pt slip")
    print(f"  Position     : {cfg['NUM_LOTS']} x {cfg['LOT_SIZE']}")
    print(f"  Date range   : {cfg['DATE_FROM']} to {cfg['DATE_TO']}")
    print(f"  Expiry       : code {cfg['EXPIRY_CODE']}")
    print("=" * 70)

    # 1. Load
    print("\n[1/5] Loading data...")
    spot, spot_raw, opts = load_data(cfg)

    # 2. Backtest
    print("\n[2/5] Running backtest...")
    trades_df = run_backtest(spot, spot_raw, opts, cfg)
    print(f"  {len(trades_df)} trades generated")

    if trades_df.empty:
        print("  No trades. Exiting.")
        return

    # 3. Stats
    print("\n[3/5] Computing statistics...")
    stats = compute_stats(trades_df, cfg)
    # Print key metrics
    print(f"  Trades       : {stats['total_trades']}")
    print(f"  Win Rate     : {stats['win_rate']}%")
    print(f"  Total PnL    : {stats['total_pnl_pts']:.1f} pts  /  Rs.{stats['total_pnl_inr']:,.0f}")
    print(f"  Profit Factor: {stats['profit_factor']}")
    print(f"  R:R          : {stats['rr_ratio']}")
    print(f"  Max DD       : {stats['max_dd_pts']:.1f} pts  /  Rs.{stats['max_dd_inr']:,.0f}")
    print(f"  Exits        : {stats['exit_counts']}")

    # 4. Save CSV
    print("\n[4/5] Saving trades CSV...")
    csv_path = os.path.join(out, 'trades.csv')
    trades_df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    # 5. Charts
    print("\n[5/5] Generating charts...")

    print("  Equity + Drawdown...")
    plot_equity_drawdown(trades_df, stats, cfg,
                         os.path.join(out, 'equity_drawdown.png'))

    print("  Monthly Heatmap...")
    plot_monthly_heatmap(trades_df, cfg,
                         os.path.join(out, 'monthly_heatmap.png'))

    print("  Day-of-Week...")
    plot_dow_breakdown(trades_df, cfg,
                       os.path.join(out, 'day_of_week.png'))

    print("  P&L Distribution + Exits...")
    plot_distribution_and_exits(trades_df, cfg,
                                os.path.join(out, 'distribution_exits.png'))

    print("  Summary Table...")
    plot_summary_table(stats, cfg,
                       os.path.join(out, 'summary.png'))

    print("  Yearly Bars...")
    plot_yearly_bars(trades_df, cfg,
                     os.path.join(out, 'yearly.png'))

    print("  Per-Trade Charts...")
    plot_trade_pages(trades_df, spot, spot_raw, opts, cfg,
                     os.path.join(out, 'trade_charts'))

    print(f"\n{'=' * 70}")
    print(f"  DONE  |  {len(trades_df)} trades  |  Results: {os.path.abspath(out)}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
