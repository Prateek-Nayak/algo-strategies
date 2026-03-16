"""
Nifty Options Selling Bot — Traffic Light Strategy
====================================================
Setup:    5MIN_INTRADAY
Exit:     TRAIL (move SL to breakeven after 1R profit)
OPT_SL:  30 pts
Capital:  ₹2,25,000 | Lot Size: 65

RULES:
- Find Red-Green or Green-Red pair on 5-min chart
- Mark range (combined high/low of pair including wicks)
- Skip pairs with range > 100 pts
- Breakout: candle CLOSES above range → SELL PE (bullish)
- Breakdown: candle CLOSES below range → SELL CE (bearish)
- Entry at NEXT candle's option OPEN (no look-ahead)
- SL: opposite side of range (spot-based)
- Trail: after 1R profit, move SL to breakeven
- OPT_SL: 30 pts option premium stop
- 3-SL rule: 3 consecutive SL → stop for the day
- Square off at 15:20
- After trade exits, scan for NEW pair from next candle onwards
"""

import pandas as pd
import numpy as np
import os, json, gc, sys, warnings
import time as clock
from datetime import datetime, time, timedelta, date

warnings.filterwarnings('ignore')

# ═══════════════ FIXED PARAMETERS ═══════════════
LOT_SIZE = 65
INITIAL_CAPITAL = 225000
STRIKE_INTERVAL = 50
EXPIRY_CODE = 1
SQUARE_OFF = time(15, 20)
MAX_RANGE = 100      # skip pairs > 100 pts
OPT_SL = 30          # option premium stop loss
MIN_PREM = 3         # minimum option premium to enter
EXIT_MODE = 'TRAIL'  # TRAIL = move SL to breakeven after 1R

def log(msg, indent=0):
    print("  " * indent + msg, flush=True)


# ═══════════════ DATA LOADING ═══════════════
def load_tree(d, usecols=None, filt=None):
    frames, n = [], 0
    for y in sorted(os.listdir(d)):
        yp = os.path.join(d, y)
        if not os.path.isdir(yp): continue
        for m in sorted(os.listdir(yp)):
            mp = os.path.join(yp, m)
            if not os.path.isdir(mp): continue
            for f in sorted(os.listdir(mp)):
                if not f.endswith('.csv'): continue
                try:
                    df = pd.read_csv(os.path.join(mp, f), usecols=usecols)
                    if filt: df = filt(df)
                    if not df.empty: frames.append(df)
                    n += 1
                except: pass
    log(f"Read {n} files", 1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_spot(bp):
    df = load_tree(os.path.join(bp, 'spot_5min'))
    df['datetime_ist'] = pd.to_datetime(df['datetime_ist'], utc=True).dt.tz_convert('Asia/Kolkata')
    return df.sort_values('datetime_ist').reset_index(drop=True)

def load_options(bp):
    needed = ['datetime_ist', 'open', 'high', 'low', 'close', 'volume', 'oi', 'iv',
              'actual_strike', 'expiry_code', 'option_type']
    df = load_tree(os.path.join(bp, 'options_data'), usecols=lambda c: c in needed,
                   filt=lambda d: d[d['expiry_code'] == EXPIRY_CODE] if 'expiry_code' in d.columns else d)
    df['datetime_ist'] = pd.to_datetime(df['datetime_ist'], utc=True).dt.tz_convert('Asia/Kolkata')
    return df.sort_values('datetime_ist').reset_index(drop=True)

def build_index(opts):
    log("Building option index...", 1)
    t0 = clock.time()
    ts = opts['datetime_ist'].astype(str).values
    stk = opts['actual_strike'].values.astype(int)
    typ = opts['option_type'].values
    cl = opts['close'].values.astype(float)
    op = opts['open'].values.astype(float) if 'open' in opts.columns else cl.copy()
    hi = opts['high'].values.astype(float) if 'high' in opts.columns else cl.copy()
    lo = opts['low'].values.astype(float) if 'low' in opts.columns else cl.copy()
    iv = opts['iv'].values.astype(float) if 'iv' in opts.columns else np.zeros(len(opts))
    idx = {}
    for i in range(len(opts)):
        idx[(ts[i], stk[i], typ[i])] = (cl[i], op[i], hi[i], lo[i], iv[i])
    log(f"Index: {len(idx):,} entries in {clock.time() - t0:.1f}s", 1)
    return idx

def get_opt(idx, ts, spot, otype):
    atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
    for s in [atm, atm + STRIKE_INTERVAL, atm - STRIKE_INTERVAL]:
        k = (ts, s, otype)
        if k in idx:
            c, o, h, l, iv = idx[k]
            return {'close': c, 'open': o, 'high': h, 'low': l, 'iv': iv, 'actual_strike': s}
    return None

def get_opt_strike(idx, ts, strike, otype):
    k = (ts, int(strike), otype)
    if k in idx:
        c, o, h, l, iv = idx[k]
        return {'close': c, 'open': o, 'high': h, 'low': l, 'iv': iv, 'actual_strike': int(strike)}
    return None


# ═══════════════ BACKTESTER ═══════════════
def backtest(spot_5min, opt_idx, start_date=None, end_date=None):
    """
    Backtest Traffic Light strategy: 5MIN_INTRADAY | TRAIL | SL=30

    Entry:  option OPEN at ci+1 (next candle after breakout confirmation)
    Exit:   option CLOSE at ci (current candle when SL/target detected)
    """
    df = spot_5min.copy()
    df['date'] = df['datetime_ist'].dt.date
    df['ct'] = df['datetime_ist'].dt.time
    df['dt_str'] = df['datetime_ist'].astype(str)

    dates = sorted(df['date'].unique())
    if start_date: dates = [d for d in dates if d >= start_date]
    if end_date: dates = [d for d in dates if d <= end_date]

    trades, daily_pnl = [], []
    capital = INITIAL_CAPITAL

    for day in dates:
        day_df = df[df['date'] == day].sort_values('datetime_ist').reset_index(drop=True)

        if len(day_df) < 3:
            daily_pnl.append({'date': str(day), 'pnl': 0, 'capital': round(capital, 2), 'num_trades': 0})
            continue

        day_pnl_v = 0
        day_trades_count = 0
        consec_sl = 0
        trade = None
        pair_scan_from = 0
        current_pair = None

        def get_opt_ts(candle_idx):
            if candle_idx >= len(day_df): return None, None
            r = day_df.iloc[candle_idx]
            return r['dt_str'], r['close']

        def get_next_candle_opt_ts(candle_idx):
            return get_opt_ts(candle_idx + 1)

        for ci in range(len(day_df)):
            row = day_df.iloc[ci]
            ct = row['ct']
            spot_c = row['close']
            spot_h = row['high']
            spot_l = row['low']

            ts_exit, _ = get_opt_ts(ci)
            if ts_exit is None: continue

            # ── 3-SL RULE ──
            if consec_sl >= 3: break

            # ── EXIT ──
            if trade is not None:
                # Square off
                if ct >= SQUARE_OFF:
                    co = get_opt_strike(opt_idx, ts_exit, trade['strike'], trade['option_type'])
                    ep = co['close'] if co else trade['entry_price']
                    pnl = trade['entry_price'] - ep; pv = pnl * LOT_SIZE
                    trade.update({'exit_price': round(ep, 2), 'exit_time': ts_exit,
                        'exit_spot': spot_c, 'pnl_points': round(pnl, 2),
                        'pnl_value': round(pv, 2), 'exit_reason': 'SQUARE_OFF'})
                    trades.append(trade); day_pnl_v += pv
                    if pnl < 0: consec_sl += 1
                    else: consec_sl = 0
                    pair_scan_from = ci + 1; current_pair = None
                    trade = None; continue

                co = get_opt_strike(opt_idx, ts_exit, trade['strike'], trade['option_type'])
                if co:
                    # Spot-based SL
                    sl_hit = False
                    if trade['dir'] == 'BULL' and spot_l <= trade['sl_spot']: sl_hit = True
                    if trade['dir'] == 'BEAR' and spot_h >= trade['sl_spot']: sl_hit = True
                    if sl_hit:
                        ep = co['close']; pnl = trade['entry_price'] - ep; pv = pnl * LOT_SIZE
                        trade.update({'exit_price': round(ep, 2), 'exit_time': ts_exit,
                            'exit_spot': spot_c, 'pnl_points': round(pnl, 2),
                            'pnl_value': round(pv, 2), 'exit_reason': 'SL_HIT'})
                        trades.append(trade); day_pnl_v += pv
                        consec_sl += 1
                        pair_scan_from = ci + 1; current_pair = None
                        trade = None; continue

                    # Option premium SL
                    if co['high'] >= trade['entry_price'] + OPT_SL:
                        ep = trade['entry_price'] + OPT_SL
                        pnl = trade['entry_price'] - ep; pv = pnl * LOT_SIZE
                        trade.update({'exit_price': round(ep, 2), 'exit_time': ts_exit,
                            'exit_spot': spot_c, 'pnl_points': round(pnl, 2),
                            'pnl_value': round(pv, 2), 'exit_reason': 'OPT_SL'})
                        trades.append(trade); day_pnl_v += pv
                        consec_sl += 1
                        pair_scan_from = ci + 1; current_pair = None
                        trade = None; continue

                    # TRAIL: move SL to breakeven after 1R profit
                    risk = trade['pair_range']
                    if trade['dir'] == 'BULL' and spot_h >= trade['entry_spot'] + risk:
                        trade['sl_spot'] = trade['entry_spot']
                        trade['trailing_active'] = True
                    if trade['dir'] == 'BEAR' and spot_l <= trade['entry_spot'] - risk:
                        trade['sl_spot'] = trade['entry_spot']
                        trade['trailing_active'] = True

            # ── FIND PAIR & ENTRY ──
            if trade is None and ct < time(15, 0):

                # Step 1: Scan for RG/GR pair
                if current_pair is None and ci >= pair_scan_from:
                    if ci >= pair_scan_from + 1:
                        c_prev = day_df.iloc[ci - 1]
                        c_curr = day_df.iloc[ci]

                        is_rg = c_prev['close'] < c_prev['open'] and c_curr['close'] > c_curr['open']
                        is_gr = c_prev['close'] > c_prev['open'] and c_curr['close'] < c_curr['open']

                        if is_rg or is_gr:
                            ph = max(c_prev['high'], c_curr['high'])
                            pl = min(c_prev['low'], c_curr['low'])
                            pr = ph - pl
                            if pr <= MAX_RANGE:
                                current_pair = {
                                    'idx1': ci - 1, 'idx2': ci,
                                    'pair_high': ph, 'pair_low': pl,
                                    'pair_range': round(pr, 2),
                                    'type': 'RG' if is_rg else 'GR',
                                }

                # Step 2: Check breakout (candle CLOSES above/below range)
                elif current_pair is not None and ci > current_pair['idx2']:
                    p = current_pair

                    breakout_dir = None
                    if spot_c >= p['pair_high']:
                        breakout_dir = 'BULL'
                    elif spot_c <= p['pair_low']:
                        breakout_dir = 'BEAR'

                    if breakout_dir:
                        ts_entry, spot_entry = get_next_candle_opt_ts(ci)
                        if ts_entry is None:
                            current_pair = None; continue

                        otype = 'PE' if breakout_dir == 'BULL' else 'CE'
                        od = get_opt(opt_idx, ts_entry, spot_entry, otype)

                        if od and od['open'] > MIN_PREM:
                            c1_row = day_df.iloc[p['idx1']]
                            c2_row = day_df.iloc[p['idx2']]
                            next_row = day_df.iloc[ci + 1] if ci + 1 < len(day_df) else row
                            trade = {
                                'type': f'SELL_{otype}',
                                'pattern': 'TL_5MIN_INTRADAY',
                                'entry_price': round(od['open'], 2),
                                'entry_time': ts_entry,
                                'entry_spot': next_row['open'],
                                'strike': od['actual_strike'],
                                'option_type': otype,
                                'dir': breakout_dir,
                                'sl_spot': p['pair_low'] if breakout_dir == 'BULL' else p['pair_high'],
                                'pair_high': p['pair_high'],
                                'pair_low': p['pair_low'],
                                'pair_range': p['pair_range'],
                                'pair_type': p['type'],
                                'pair_c1_time': str(c1_row.get('datetime_ist', '')),
                                'pair_c2_time': str(c2_row.get('datetime_ist', '')),
                                'date': str(day),
                                'setup': '5MIN_INTRADAY',
                                'exit_mode': 'TRAIL',
                                'trailing_active': False,
                            }
                            day_trades_count += 1
                            current_pair = None
                            continue
                        else:
                            current_pair = None

        # EOD close any open trade
        if trade is not None:
            lr = day_df.iloc[-1]
            ts_eod, _ = get_opt_ts(len(day_df) - 1)
            if ts_eod is None:
                ts_eod = lr['dt_str']
            co = get_opt_strike(opt_idx, ts_eod, trade['strike'], trade['option_type'])
            ep = co['close'] if co else trade['entry_price']
            pnl = trade['entry_price'] - ep; pv = pnl * LOT_SIZE
            trade.update({'exit_price': round(ep, 2), 'exit_time': ts_eod,
                'exit_spot': lr['close'], 'pnl_points': round(pnl, 2),
                'pnl_value': round(pv, 2), 'exit_reason': 'SQUARE_OFF'})
            trades.append(trade); day_pnl_v += pv

        capital += day_pnl_v
        daily_pnl.append({'date': str(day), 'pnl': round(day_pnl_v, 2),
            'capital': round(capital, 2), 'num_trades': day_trades_count})

    equity = [{'datetime': d['date'], 'capital': d['capital']} for d in daily_pnl]
    return trades, daily_pnl, equity


# ═══════════════ STATS ═══════════════
def calc_stats(trades, daily, label):
    tot = sum(d['pnl'] for d in daily)
    n = len(trades)
    if n == 0:
        return {'label': label, 'total_pnl': 0, 'num_trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0, 'avg_win_points': 0, 'avg_loss_points': 0,
                'profit_factor': 0, 'max_drawdown_pct': 0, 'return_pct': 0,
                'sharpe': 0, 'exit_reasons': {}, 'signal_days': 0, 'trading_days': len(daily),
                'ce_trades': 0, 'pe_trades': 0}

    w = [t for t in trades if t['pnl_points'] > 0]
    l = [t for t in trades if t['pnl_points'] <= 0]

    pk, mdd = INITIAL_CAPITAL, 0
    for d in daily:
        pk = max(pk, d['capital'])
        mdd = max(mdd, (pk - d['capital']) / pk)

    gp = sum(t['pnl_value'] for t in w) if w else 0
    gl = abs(sum(t['pnl_value'] for t in l)) if l else 1

    exits = {}
    for t in trades:
        r = t.get('exit_reason', '?')
        exits[r] = exits.get(r, 0) + 1

    dv = [d['pnl'] for d in daily if d['num_trades'] > 0]

    return {
        'label': label,
        'total_pnl': round(tot, 2),
        'num_trades': n,
        'wins': len(w),
        'losses': len(l),
        'win_rate': round(len(w) / n * 100, 1),
        'avg_win_points': round(np.mean([t['pnl_points'] for t in w]), 2) if w else 0,
        'avg_loss_points': round(np.mean([t['pnl_points'] for t in l]), 2) if l else 0,
        'profit_factor': round(gp / (gl + 1e-10), 2),
        'max_drawdown_pct': round(mdd * 100, 2),
        'return_pct': round(tot / INITIAL_CAPITAL * 100, 2),
        'sharpe': round(np.mean(dv) / (np.std(dv) + 1e-10) * np.sqrt(252), 2) if len(dv) > 1 else 0,
        'exit_reasons': exits,
        'trading_days': len(daily),
        'signal_days': sum(1 for d in daily if d['num_trades'] > 0),
        'ce_trades': sum(1 for t in trades if t['type'] == 'SELL_CE'),
        'pe_trades': sum(1 for t in trades if t['type'] == 'SELL_PE'),
    }


def print_stats(s):
    print(f"\n  {'─' * 60}")
    print(f"  {s['label']}")
    print(f"  {'─' * 60}")
    print(f"  PnL:          ₹{s['total_pnl']:>12,.0f}  ({s['return_pct']}%)")
    print(f"  Trades:        {s['num_trades']:>6} (W:{s['wins']} L:{s['losses']})")
    print(f"  Signal Days:   {s['signal_days']:>6} / {s['trading_days']}")
    print(f"  Win Rate:      {s['win_rate']:>6}%")
    print(f"  Avg Win:       {s['avg_win_points']:>6.1f} pts | Avg Loss: {s['avg_loss_points']:.1f} pts")
    print(f"  Profit Factor: {s['profit_factor']:>6}")
    print(f"  Max Drawdown:  {s['max_drawdown_pct']:>6}%")
    print(f"  Sharpe:        {s['sharpe']:>6}")
    print(f"  CE/PE:         {s['ce_trades']}/{s['pe_trades']}")
    print(f"  Exits:         {s['exit_reasons']}")


# ═══════════════ MAIN ═══════════════
def run(bp):
    T0 = clock.time()
    print("=" * 70)
    print("  ⚡ NIFTY OPTIONS BOT — TRAFFIC LIGHT STRATEGY")
    print("  5MIN_INTRADAY | TRAIL | OPT_SL=30")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,} | Lot: {LOT_SIZE}")
    print("=" * 70)

    log("\n[1/3] Loading data...")
    spot = load_spot(bp)
    log(f"{len(spot):,} candles", 1)
    opts = load_options(bp)
    log(f"{len(opts):,} records", 1)
    log("Building index...")
    idx = build_index(opts)
    del opts; gc.collect()

    train_start, train_end = date(2022, 1, 1), date(2024, 12, 31)
    test_start, test_end = date(2025, 1, 1), date(2026, 12, 31)

    # ── TRAIN ──
    log(f"\n[2/3] Backtesting TRAIN ({train_start} → {train_end})...")
    tr_trades, tr_daily, tr_equity = backtest(spot, idx, train_start, train_end)
    tr_stats = calc_stats(tr_trades, tr_daily, "TRAIN (2022-2024)")
    print_stats(tr_stats)

    # ── TEST ──
    log(f"\n[3/3] Backtesting TEST ({test_start} → {test_end})...")
    te_trades, te_daily, te_equity = backtest(spot, idx, test_start, test_end)
    te_stats = calc_stats(te_trades, te_daily, "TEST (2025-2026)")
    print_stats(te_stats)

    # ── EXPORT ──
    log("\nExporting results...")
    dc = spot.groupby(spot['datetime_ist'].dt.date).agg(
        open=('open', 'first'), high=('high', 'max'),
        low=('low', 'min'), close=('close', 'last')).reset_index()
    dc.columns = ['date', 'open', 'high', 'low', 'close']
    dc['date'] = dc['date'].astype(str)

    results = {
        'params': {
            'setup': '5MIN_INTRADAY', 'exit_mode': 'TRAIL', 'opt_sl': OPT_SL,
            'max_range': MAX_RANGE, 'min_prem': MIN_PREM,
            'strategy': 'Traffic Light - Red/Green Pair Breakout (Close Confirmation)',
        },
        'train_stats': tr_stats,
        'test_stats': te_stats,
        'train_trades': tr_trades,
        'test_trades': te_trades,
        'train_daily_pnl': tr_daily,
        'test_daily_pnl': te_daily,
        'train_equity': tr_equity,
        'test_equity': te_equity,
        'daily_candles': dc.to_dict('records'),
        'config': {
            'lot_size': LOT_SIZE, 'initial_capital': INITIAL_CAPITAL,
            'strike_interval': STRIKE_INTERVAL,
            'strategy': 'Traffic Light - 5MIN INTRADAY | TRAIL | SL=30',
        },
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(out, 'w') as f:
        json.dump(results, f, default=str)

    total = clock.time() - T0
    print(f"\n{'=' * 70}")
    print(f"  ✅ DONE in {total / 60:.1f} min → {out}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    bp = sys.argv[1] if len(sys.argv) > 1 else input("Base path: ").strip()
    run(bp)