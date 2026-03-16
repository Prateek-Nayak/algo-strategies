"""
Supertrend Calculator — TradingView Compatible
===============================================
Extracted from claude_st_logger.py for live use in index.py.

Provides:
  - calculate_supertrend_tradingview(df, period, multiplier)  — batch on DataFrame
  - SupertrendLive(period, multiplier)                        — incremental candle-by-candle
"""

import pandas as pd
import numpy as np
from collections import deque


# ── Batch (DataFrame) calculator ────────────────────────────────────────────

def calculate_atr_rma(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """ATR via Wilder's RMA (matches TradingView)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = pd.Series(index=df.index, dtype=float)
    atr.iloc[period - 1] = true_range.iloc[:period].mean()
    for i in range(period, len(df)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + true_range.iloc[i]) / period
    return atr


def calculate_supertrend_tradingview(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Batch Supertrend on a DataFrame (same logic as claude_st_logger.py)."""
    df = df.copy()
    df['atr'] = calculate_atr_rma(df, period)
    hl2 = (df['high'] + df['low']) / 2
    df['basic_ub'] = hl2 + (multiplier * df['atr'])
    df['basic_lb'] = hl2 - (multiplier * df['atr'])
    df['final_ub'] = 0.0
    df['final_lb'] = 0.0
    df['supertrend'] = 0.0
    df['trend'] = True

    for i in range(len(df)):
        if pd.isna(df['atr'].iloc[i]):
            df.loc[df.index[i], 'final_ub'] = np.nan
            df.loc[df.index[i], 'final_lb'] = np.nan
            df.loc[df.index[i], 'supertrend'] = np.nan
            df.loc[df.index[i], 'trend'] = True
            continue

        if i == 0 or pd.isna(df['atr'].iloc[i - 1]):
            df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
            df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
            df.loc[df.index[i], 'supertrend'] = df['final_ub'].iloc[i]
            df.loc[df.index[i], 'trend'] = True
        else:
            if (df['basic_ub'].iloc[i] < df['final_ub'].iloc[i - 1]
                    or df['close'].iloc[i - 1] > df['final_ub'].iloc[i - 1]):
                df.loc[df.index[i], 'final_ub'] = df['basic_ub'].iloc[i]
            else:
                df.loc[df.index[i], 'final_ub'] = df['final_ub'].iloc[i - 1]

            if (df['basic_lb'].iloc[i] > df['final_lb'].iloc[i - 1]
                    or df['close'].iloc[i - 1] < df['final_lb'].iloc[i - 1]):
                df.loc[df.index[i], 'final_lb'] = df['basic_lb'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lb'] = df['final_lb'].iloc[i - 1]

            if df['trend'].iloc[i - 1]:
                if df['close'].iloc[i] <= df['final_lb'].iloc[i]:
                    df.loc[df.index[i], 'trend'] = False
                    df.loc[df.index[i], 'supertrend'] = df['final_ub'].iloc[i]
                else:
                    df.loc[df.index[i], 'trend'] = True
                    df.loc[df.index[i], 'supertrend'] = df['final_lb'].iloc[i]
            else:
                if df['close'].iloc[i] >= df['final_ub'].iloc[i]:
                    df.loc[df.index[i], 'trend'] = True
                    df.loc[df.index[i], 'supertrend'] = df['final_lb'].iloc[i]
                else:
                    df.loc[df.index[i], 'trend'] = False
                    df.loc[df.index[i], 'supertrend'] = df['final_ub'].iloc[i]

    df['trend_direction'] = df['trend'].apply(lambda x: 1 if x else -1)
    return df


# ── Incremental (live) calculator ───────────────────────────────────────────

class SupertrendLive:
    """
    Candle-by-candle Supertrend calculator.
    Maintains internal state so each new candle is O(1) — no recompute.

    Usage:
        st = SupertrendLive(period=10, multiplier=3.0)
        st.warm_up(historical_df)          # seed with 50+ candles
        direction, value = st.update(o, h, l, c)  # feed each new candle
        direction, value = st.current()    # read current state anytime
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.reset()

    def reset(self):
        """Clear all internal state."""
        self._warmed = False
        self._candle_count = 0
        self._prev_close = None

        # ATR RMA state
        self._atr = None
        self._tr_buffer = deque(maxlen=self.period)  # for initial SMA

        # Band state
        self._final_ub = None
        self._final_lb = None
        self._trend = True  # True = uptrend (1), False = downtrend (-1)
        self._supertrend = None

    @property
    def is_ready(self) -> bool:
        return self._warmed and self._supertrend is not None

    def current(self):
        """Return (direction, supertrend_value). direction: 1=bull, -1=bear."""
        if not self.is_ready:
            return None, None
        return (1 if self._trend else -1), self._supertrend

    def warm_up(self, df: pd.DataFrame):
        """
        Seed with historical OHLC DataFrame (must have 'open','high','low','close').
        Processes all rows to establish ATR + band state.
        """
        self.reset()
        required = ['open', 'high', 'low', 'close']
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for r in required:
            if r not in cols_lower:
                raise ValueError(f"Column '{r}' missing in warmup data. Found: {list(df.columns)}")

        for i in range(len(df)):
            row = df.iloc[i]
            o = float(row[cols_lower['open']])
            h = float(row[cols_lower['high']])
            l = float(row[cols_lower['low']])
            c = float(row[cols_lower['close']])
            self.update(o, h, l, c)

        self._warmed = True

    def update(self, o: float, h: float, l: float, c: float):
        """
        Feed one completed candle. Returns (direction, supertrend_value).
        direction: 1 = bullish, -1 = bearish, None = not ready yet.
        """
        self._candle_count += 1

        # ── True Range ──
        if self._prev_close is not None:
            tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        else:
            tr = h - l

        self._tr_buffer.append(tr)

        # ── ATR (RMA) ──
        if self._candle_count < self.period:
            self._prev_close = c
            return None, None

        if self._candle_count == self.period:
            # First ATR = SMA of first `period` true ranges
            self._atr = sum(self._tr_buffer) / self.period
        else:
            # Wilder's RMA
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        # ── Basic bands ──
        hl2 = (h + l) / 2
        basic_ub = hl2 + self.multiplier * self._atr
        basic_lb = hl2 - self.multiplier * self._atr

        # ── Final bands ──
        if self._final_ub is None:
            # Very first valid bar
            self._final_ub = basic_ub
            self._final_lb = basic_lb
            self._supertrend = basic_ub
            self._trend = True
        else:
            # Final Upper Band
            if basic_ub < self._final_ub or self._prev_close > self._final_ub:
                new_ub = basic_ub
            else:
                new_ub = self._final_ub

            # Final Lower Band
            if basic_lb > self._final_lb or self._prev_close < self._final_lb:
                new_lb = basic_lb
            else:
                new_lb = self._final_lb

            self._final_ub = new_ub
            self._final_lb = new_lb

            # ── Trend decision ──
            if self._trend:  # was uptrend
                if c <= self._final_lb:
                    self._trend = False
                    self._supertrend = self._final_ub
                else:
                    self._trend = True
                    self._supertrend = self._final_lb
            else:  # was downtrend
                if c >= self._final_ub:
                    self._trend = True
                    self._supertrend = self._final_lb
                else:
                    self._trend = False
                    self._supertrend = self._final_ub

        self._prev_close = c
        direction = 1 if self._trend else -1
        return direction, round(self._supertrend, 2)
