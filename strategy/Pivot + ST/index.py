from datetime import datetime, timedelta
import math, os
import random
import sys
import threading
import platform
import pytz
import json
import signal
import time
import requests
import pyotp
import pandas as pd
from collections import defaultdict
from pathlib import Path

from tomlkit import date

# Allow importing from parent directory (strategy root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nxtradstream import NxtradStream
from supertrend import SupertrendLive
import logging
import cProfile
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import gzip
import pstats
import io
from zoneinfo import ZoneInfo
import argparse  # <-- added

load_dotenv()

# ====== MULTI-ACCOUNT LOADER (added) ========================================-
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--account", type=int, default=1, help="Account index (1..N) to run")
    return p.parse_args()

def load_account_env(idx: int):
    """
    Loads ACCOUNT_ID_{idx}, ACCOUNT_{idx}_NAME, API_KEY_{idx}, PASSWORD_{idx}, TOTP_SECRET_{idx}, LOG_DIR_{idx}
    into module-level globals so the rest of the code stays unchanged.
    """
    global ACCOUNT_ID, ACCOUNT_NAME, API_KEY, PASSWORD, TOTP_SECRET, AUTHORIZATION, LOG_DIR, ACCOUNT_INDEX

    ACCOUNT_INDEX = idx

    ACCOUNT_ID   = os.getenv(f'ACCOUNT_ID_{idx}')
    ACCOUNT_NAME = os.getenv(f'ACCOUNT_{idx}_NAME')
    API_KEY      = os.getenv(f'API_KEY_{idx}')
    PASSWORD     = os.getenv(f'PASSWORD_{idx}')
    TOTP_SECRET  = os.getenv(f'TOTP_SECRET_{idx}')
    LOG_DIR      = os.getenv(f'LOG_DIR_{idx}') or os.getenv('LOG_DIR_1') or f"logs/account_{idx}"

    missing = [k for k,v in {
        f'ACCOUNT_ID_{idx}':ACCOUNT_ID,
        f'ACCOUNT_{idx}_NAME':ACCOUNT_NAME,
        f'API_KEY_{idx}':API_KEY,
        f'PASSWORD_{idx}':PASSWORD,
        f'TOTP_SECRET_{idx}':TOTP_SECRET,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env for account index {idx}: {', '.join(missing)}")

    # refresh derived globals
    AUTHORIZATION = f"Bearer {API_KEY}"
    os.makedirs(LOG_DIR, exist_ok=True)
# ============================================================================

# Set a safe default; real value is overwritten by load_account_env(...)
LOG_DIR = os.getenv('LOG_DIR_1') or "logs"
os.makedirs(LOG_DIR, exist_ok=True)
_EOD_EMAIL_SENT_FOR_DATE = None  # prevent duplicates per day

# --- Holiday cache (used by expiry + trading-day checks) ---
_HOLIDAY_CACHE_DATE = None       # datetime.date (IST) for which cache is valid
_NFO_CLOSED_DATES = set()        # set of 'YYYY-MM-DD' strings

NIFTY_SYMBOL_ID_FOR_OHLC = "IDX_-1_NSE"
NIFTY_SYMBOL = "-1_NSE"
INTERVAL = 5
LOT_SIZE = 65
DATA_OUTPUT_DIR = "data"
DATE_FORMAT = "%Y-%m-%d"
FILENAME_TEMPLATE = "nifty_{date}.csv"
LOGS_OUTPUT_DIR = Path("logs/")  # Update with your actual output directory
CANDLES_OUTPUT_PATH = LOGS_OUTPUT_DIR / "candles/"
CANDLES_FILENAME_TEMPLATE = "{date}_candles.csv"
STREAM_CANDLES_FILENAME_TEMPLATE = "{date}_stream_candles.csv"

RUNNING = True
ORDER_TYPE = "bracket"  # "normal" or "bracket"
ORDER_PLACED = False
CURRENT_ORDER_ID = None
EXPIRY_DATE = None
ohlc_data = defaultdict(lambda: {
    "open": None, "high": float("-inf"), "low": float("inf"), "close": None
})

# === Supertrend state ===
SPOT_DATA_DIR = "data/spot_5min"  # cached spot data for ST warmup
_nifty_st = SupertrendLive(period=10, multiplier=3.0)
_option_st = SupertrendLive(period=10, multiplier=3.0)
_pending_pivot_trigger = None   # set when candle i touches pivot
_active_option_symbol = None    # symId of sold option (for OHLC subscription)
_entry_direction = None         # "PE" or "CE" — current position direction
_entry_st_direction = None      # 1 (bullish) or -1 (bearish) at entry

# OHLC 5-min aggregation (stream sends 1-min updates, we aggregate into 5-min blocks)
_nifty_5m = {"block": None, "open": None, "high": None, "low": None, "close": None}
_option_5m = {"block": None, "open": None, "high": None, "low": None, "close": None}

def _ohlc_time_to_5m_block(candle_time_str):
    """Parse OHLC candle time string and return the 5-min block start as a string."""
    try:
        # candle_time comes as e.g. '2026-03-04 10:19:00'
        dt = datetime.strptime(candle_time_str.strip(), "%Y-%m-%d %H:%M:%S")
        block = dt.replace(second=0, microsecond=0, minute=(dt.minute // 5) * 5)
        return block.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return candle_time_str  # fallback

from dotenv import dotenv_values

# Email config (read from .env.master)
_master_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env.master")
_master_env = dotenv_values(_master_env_path)

SMTP_HOST = _master_env.get("SMTP_HOST", os.getenv("SMTP_HOST", ""))
SMTP_PORT = int(_master_env.get("SMTP_PORT", os.getenv("SMTP_PORT", "587")))
SMTP_USER = _master_env.get("SMTP_USER", os.getenv("SMTP_USER", ""))
SMTP_PASS = _master_env.get("SMTP_PASSWORD", os.getenv("SMTP_PASS", ""))
MAIL_FROM = _master_env.get("SMTP_USER", os.getenv("MAIL_FROM", ""))
MAIL_TO   = _master_env.get("EMAIL_TO", os.getenv("MAIL_TO", ""))

API_HOST = "api.tradejini.com"
ACCESS_TOKEN_URL = f"https://{API_HOST}/v2/api-gw/oauth/individual-token-v2"
CHART_URL = f"https://{API_HOST}/v2/api/mkt-data/chart/interval-data"
PLACE_ORDER_URL = f"https://{API_HOST}/v2/api/oms/place-order"
BRACKET_ORDER_URL = f"https://{API_HOST}/v2/api/oms/place-order/bo"
ORDER_HISTORY_URL = f"https://{API_HOST}/v2/api/oms/history"
MARGIN_URL = f"https://{API_HOST}/v2/api/oms/margin"
LIMITS_URL = f"https://{API_HOST}/v2/api/oms/limits"
GET_ORDER_DETAILS_URL = f"https://{API_HOST}/v2/api/oms/orders"
PLACE_OCO_ORDER_URL = f"https://{API_HOST}/v2/api/oms/place-order/oco"
EXIT_ORDER_URL = f"https://{API_HOST}/v2/api/oms/exit-order"
MODIFY_ORDER_URL = f"https://{API_HOST}/v2/api/oms/modify-order"

# NOTE: AUTHORIZATION is set inside load_account_env(...)
# AUTHORIZATION = f"Bearer {API_KEY}"   # <-- removed to avoid NameError before env load

# Reusable HTTP session for API calls (connection pooling / keep-alive)
_api_session = requests.Session()
_api_session.headers.update({"Accept": "application/json"})

# === Timezone helpers ===
IST = pytz.timezone("Asia/Kolkata")

def ist_now():
    return datetime.now(IST)

def in_window(now_ist, start_hm=("09","00"), end_hm=("15","30")):
    s_h, s_m = map(int, start_hm)
    e_h, e_m = map(int, end_hm)
    start = now_ist.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
    end   = now_ist.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
    return start <= now_ist < end

def send_email(subject: str, body: str,
               attachment_name: str = None,
               attachment_bytes: bytes = None,
               html: bool = False):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM and MAIL_TO):
        logger.info("✉️ Email not configured; skipping send.")
        return False
    
    # --- JITTER: small random delay so 20 instances don't all send at once ---
    try:
        delay = random.uniform(0, 2) 
        logger.info(f"⏱️ Email jitter delay: {delay:.2f}s before sending.")
        time.sleep(delay)
    except Exception:
        pass

    _mime_sub = "html" if html else "plain"

    try:
        if not attachment_bytes:
            msg = MIMEText(body, _mime_sub, "utf-8")
        else:
            msg = MIMEMultipart()
            msg.attach(MIMEText(body, _mime_sub, "utf-8"))

            if isinstance(attachment_bytes, str):
                size = os.path.getsize(attachment_bytes)
                if size > 20 * 1024 * 1024:
                    body += f"\n\n⚠️ Log file too large ({size//1024//1024} MB). Saved at {attachment_bytes}."
                    msg = MIMEText(body, _mime_sub, "utf-8")
                else:
                    with open(attachment_bytes, "rb") as f:
                        part = MIMEApplication(f.read(), _subtype="gzip")
                    fname = attachment_name or os.path.basename(attachment_bytes)
                    part.add_header("Content-Disposition", "attachment", filename=fname)
                    msg.attach(part)
            else:
                part = MIMEApplication(attachment_bytes, _subtype="gzip")
                part.add_header("Content-Disposition", "attachment", filename=attachment_name or "attachment.txt.gz")
                msg.attach(part)

        msg["Subject"] = f"{ACCOUNT_ID} : {ACCOUNT_NAME} " + subject
        msg["From"] = MAIL_FROM
        msg["To"] = MAIL_TO
        msg["Date"] = formatdate(localtime=True)
        MAX_RETRIES = 3
        BASE_BACKOFF = 30

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"✉️ Attempt {attempt}/{MAX_RETRIES} to send email.")
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
                    s.starttls()
                    s.login(SMTP_USER, SMTP_PASS)
                    s.sendmail(MAIL_FROM, [MAIL_TO], msg.as_string())
                logger.info("✉️ Email sent.")
                return True

            except smtplib.SMTPResponseException as smtp_err:
                code = smtp_err.smtp_code
                err_msg = smtp_err.smtp_error
                logger.info(f"❌ Email send failed (SMTP {code}): {err_msg}")
                if 400 <= code < 500 and attempt < MAX_RETRIES:
                    sleep_for = BASE_BACKOFF * attempt
                    logger.info(f"⏳ Temporary SMTP error {code}. Retrying in {sleep_for}s.")
                    time.sleep(sleep_for)
                    continue
                return False

            except Exception as e:
                logger.info(f"❌ Email send failed on attempt {attempt}: {e}")
                if attempt < MAX_RETRIES:
                    sleep_for = BASE_BACKOFF * attempt
                    logger.info(f"⏳ Will retry in {sleep_for}s.")
                    time.sleep(sleep_for)
                    continue
                return False

    except Exception as e:
        logger.info(f"❌ Email send failed (setup/build error): {e}")
        return False

def send_email_async(subject, body, html=False, attachment_name=None, attachment_bytes=None):
    """Non-blocking email send via background thread."""
    t = threading.Thread(target=send_email, args=(subject, body),
                         kwargs={"html": html, "attachment_name": attachment_name,
                                 "attachment_bytes": attachment_bytes},
                         daemon=True)
    t.start()

# === HTML Email Helpers (inline-CSS only for max email-client compat) ===
def _html_badge(text, color="blue"):
    _c = {"red": ("#fce8e6", "#c5221f"), "green": ("#e6f4ea", "#137333"),
          "blue": ("#e8f0fe", "#1a73e8"), "orange": ("#fef7e0", "#e37400"),
          "gray": ("#f1f3f4", "#5f6368")}
    bg, fg = _c.get(color, _c["blue"])
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:4px;font-size:12px;font-weight:600;display:inline-block">{text}</span>')

def _html_alert(text, color="blue"):
    _bc = {"red": "#c5221f", "green": "#137333", "blue": "#1a73e8", "orange": "#e37400"}
    _bg = {"red": "#fce8e6", "green": "#e6f4ea", "blue": "#e8f0fe", "orange": "#fef7e0"}
    return (f'<table width="100%" style="background:{_bg.get(color, "#e8f0fe")};border-radius:6px;'
            f'border-left:4px solid {_bc.get(color, "#1a73e8")};margin:12px 0"><tr>'
            f'<td style="padding:12px 16px;color:#333;font-size:13px">{text}</td></tr></table>')

def _html_kv_table(pairs):
    rows = "".join(
        f'<tr><td style="padding:5px 0;color:#5f6368;width:130px;font-size:13px;vertical-align:top">{k}</td>'
        f'<td style="padding:5px 0;color:#333;font-size:13px;font-weight:500">{v}</td></tr>'
        for k, v in pairs if v is not None
    )
    return f'<table width="100%" cellpadding="0" cellspacing="0">{rows}</table>'

def _html_section(title):
    return (f'<h2 style="margin:16px 0 8px;color:#333;font-size:14px;font-weight:600;'
            f'border-bottom:2px solid #e0e0e0;padding-bottom:6px">{title}</h2>')

def _html_email_wrap(title, subtitle, body_html):
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
        f'<body style="margin:0;padding:0;background:#f4f4f7;'
        f"font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif\">"
        f'<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f7;padding:16px 8px">'
        f'<tr><td align="center">'
        f'<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;background:#fff;'
        f'border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">'
        f'<tr><td style="background:linear-gradient(135deg,#1a73e8,#0d47a1);padding:20px 24px">'
        f'<h1 style="margin:0;color:#fff;font-size:18px;font-weight:600">{title}</h1>'
        f'<p style="margin:4px 0 0;color:rgba(255,255,255,0.85);font-size:12px">{subtitle}</p></td></tr>'
        f'<tr><td style="padding:20px 24px">{body_html}</td></tr>'
        f'<tr><td style="padding:14px 24px;border-top:1px solid #e0e0e0">'
        f'<p style="margin:0;color:#9aa0a6;font-size:11px;text-align:center">'
        f"Algo Trading System · {ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}</p>"
        f'</td></tr></table></td></tr></table></body></html>'
    )

def send_order_email(event: str, info: dict, latency_ms: float = None):
    """Send a formatted HTML order notification email (non-blocking)."""
    now = ist_now().strftime('%Y-%m-%d %H:%M:%S IST')
    subtitle = f"Account: {ACCOUNT_ID} · {ACCOUNT_NAME} · {now}"
    is_fail = "FAIL" in event.upper()
    order_id = info.get('orderId', '-')
    sym_id = info.get('symId', '-')

    if is_fail:
        alert = _html_alert(
            f"<strong>⚠️ {event}</strong><br>"
            f"Order: {order_id} · Symbol: {sym_id}<br>"
            f"Error: {info.get('remarks', '-')}", "red")
    else:
        alert = _html_alert(
            f"<strong>✅ {event}</strong><br>"
            f"Order: {order_id} · Symbol: {sym_id}", "green")

    lat_html = ""
    if latency_ms is not None:
        lc = "green" if latency_ms < 1000 else ("orange" if latency_ms < 2000 else "red")
        lat_html = _html_alert(f"⏱️ Signal → Order latency: <strong>{latency_ms:.0f} ms</strong>", lc)

    side_val = info.get("side")
    side_html = _html_badge(side_val.upper(), "red" if side_val == "sell" else "green") if side_val else None
    status_val = str(info.get("status", ""))
    status_html = (_html_badge(status_val, "green" if status_val in ("filled", "completed", "200") else "red")
                   if status_val else None)

    pairs = [
        ("Order ID", order_id if order_id != '-' else None),
        ("Symbol", sym_id if sym_id != '-' else None),
        ("Product", info.get("product")),
        ("Type", info.get("type")),
        ("Side", side_html),
        ("Quantity", info.get("qty")),
        ("Avg Price", f"₹{info['avgPrice']}" if info.get("avgPrice") else None),
        ("Status", status_html),
        ("Stop Loss", f"₹{info['stopLoss']}" if info.get("stopLoss") else None),
        ("Target", f"₹{info['target']}" if info.get("target") else None),
        ("Trig Price", info.get("trigPrice")),
        ("Limit Price", info.get("limitPrice")),
        ("Leg Type", info.get("legType")),
        ("Main Leg ID", info.get("mainLegOrderId")),
        ("Remarks", info.get("remarks")),
    ]
    details = _html_kv_table([(k, v) for k, v in pairs if v is not None])
    body = alert + lat_html + _html_section("Order Details") + details
    email_html = _html_email_wrap(f"📋 {event}", subtitle, body)
    subject = f"Order {event}: {order_id if order_id != '-' else sym_id}"
    send_email_async(subject, email_html, html=True)


def send_st_flip_exit_email(nifty_st_dir, nifty_st_val, opt_dir, opt_val, exit_ms):
    """Send detailed HTML email when ST flip exit is triggered."""
    now_str = ist_now().strftime('%Y-%m-%d %H:%M:%S IST')
    subtitle = f"Account: {ACCOUNT_ID} · {ACCOUNT_NAME} · {now_str}"

    nifty_trend = "BULLISH ↑" if nifty_st_dir == 1 else "BEARISH ↓"
    opt_trend = "BULLISH ↑" if opt_dir == 1 else "BEARISH ↓"
    entry_trend = "BULLISH" if _entry_st_direction == 1 else "BEARISH"
    flip_desc = f"{entry_trend} → {nifty_trend}"

    # Alert banner
    alert = _html_alert(
        f"<strong>⚠️ SUPERTREND FLIP EXIT</strong><br>"
        f"Position <strong>SELL {_entry_direction or '-'}</strong> closed via dual ST flip<br>"
        f"Nifty ST: <strong>{flip_desc}</strong> · Option ST: <strong>{opt_trend}</strong>",
        "red" if exit_ms > 2000 else "orange"
    )

    # Details table
    pairs = [
        ("Position", f"SELL {_entry_direction or '-'}"),
        ("Order ID", str(CURRENT_ORDER_ID or '-')),
        ("Entry ST Direction", entry_trend),
        ("Current Nifty ST", f"{nifty_trend} ({nifty_st_val})"),
        ("Current Option ST", f"{opt_trend} ({opt_val})"),
        ("Exit Latency", f"{exit_ms:.0f} ms"),
        ("Exit Reason", "DUAL ST FLIP"),
        ("Time", now_str),
    ]
    details = _html_kv_table(pairs)

    # Explanation section
    explain = _html_section("Exit Logic") + (
        '<p style="color:#5f6368;font-size:13px;line-height:1.6;margin:8px 0 16px">'
        'Nifty 5-min Supertrend flipped <strong>against</strong> the entry direction, '
        'AND the sold option\'s 5-min Supertrend turned <strong>BULLISH</strong> '
        '(confirming premium is rising against the short position). '
        'Both conditions met — position exited automatically.</p>'
    )

    body = alert + _html_section("Trade Details") + details + explain
    email_html = _html_email_wrap("🚨 ST Flip Exit", subtitle, body)
    send_email_async("🚨 ST Flip Exit — Position Closed", email_html, html=True)
    logger.info("📧 ST flip exit email sent.")

def read_logs_between(start_ist: datetime, end_ist: datetime) -> str:
    """
    Return logs (timestamp-based) between start_ist and end_ist inclusive, across date boundaries.
    """
    def _paths_for_range(a: datetime, b: datetime):
        days = []
        d = a.date()
        while d <= b.date():
            days.append(d)
            d = d + timedelta(days=1)
        return [current_log_path_for_date(IST.localize(datetime.combine(d, datetime.min.time()))) for d in days]

    paths = _paths_for_range(start_ist, end_ist)
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        ts_str = line.split(" - ", 1)[0]  # "YYYY-MM-DD HH:MM:SS"
                        ts = IST.localize(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
                        if start_ist <= ts <= end_ist:
                            out.append(line.rstrip("\n"))
                    except Exception:
                        pass
        except Exception as e:
            out.append(f"[Failed reading {p}: {e}]")
    return "\n".join(out) if out else "[No logs in window]"


def send_eod_email(now_ist: datetime):
    """
    Send one EOD (End Of Day) email at/after 15:30 IST for the current date, with logs 09:00–15:30.
    Always attaches logs as gzip. Uses file path to avoid memory issues on EC2.
    """
    global _EOD_EMAIL_SENT_FOR_DATE
    trade_day = now_ist.date()
    if _EOD_EMAIL_SENT_FOR_DATE == trade_day:
        return  # already sent

    start_ist = now_ist.replace(hour=9, minute=0, second=0, microsecond=0)
    end_ist   = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    body_text = read_logs_between(start_ist, end_ist)

    subject = f" - EOD Logs {trade_day} (09:00–15:30 IST)"
    header  = f"EOD @ {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}\nWindow: 09:00–15:30 IST\n\n"

    attach_name = f"EOD_{trade_day}_0900-1530.txt.gz"
    attach_path = Path("/tmp") / attach_name

    # Write compressed logs directly to disk (no large memory usage)
    with gzip.open(attach_path, "wb") as gz:
        gz.write(body_text.encode("utf-8"))

    # Pass the file path as attachment_bytes (string) to avoid reading large file into RAM
    send_email(
        subject,
        header,
        attachment_name=attach_name,
        attachment_bytes=str(attach_path)  # safe, large file won't blow up RAM
    )

    _EOD_EMAIL_SENT_FOR_DATE = trade_day

def current_log_path_for_date(d: datetime):
    return os.path.join(LOG_DIR, f"{d.strftime(DATE_FORMAT)}.txt")

# === State reset & logging setup ===
def reset_state(reason: str):
    global ORDER_PLACED, CURRENT_ORDER_ID, ohlc_data, PIVOT_LEVELS, EXPIRY_DATE
    global _pending_pivot_trigger, _active_option_symbol, _entry_direction, _entry_st_direction
    ORDER_PLACED = False
    CURRENT_ORDER_ID = None
    EXPIRY_DATE = None
    _pending_pivot_trigger = None
    _active_option_symbol = None
    _entry_direction = None
    _entry_st_direction = None
    ohlc_data.clear()
    _nifty_st.reset()
    _option_st.reset()
    _nifty_5m.update({"block": None, "open": None, "high": None, "low": None, "close": None})
    _option_5m.update({"block": None, "open": None, "high": None, "low": None, "close": None})
    try:
        PIVOT_LEVELS.clear()
    except Exception:
        pass
    logger.info(f"🔁 State reset ({reason}) for {ACCOUNT_NAME}.")

def cleanup_old_logs(retention_days: int = 30):
    """
    Delete log files and data files older than retention_days.
    Cleans up:
    - Daily log files in LOG_DIR (*.txt)
    - Candle data in DATA_OUTPUT_DIR (nifty_*.csv)
    - Candle CSVs in CANDLES_OUTPUT_PATH (*_candles.csv)
    - Profile files (*.prof, *.profile.txt)
    """
    now_ist = ist_now()
    cutoff_date = (now_ist - timedelta(days=retention_days)).date()
    
    deleted_count = 0
    
    try:
        # 1. Clean daily log files in LOG_DIR
        for filename in os.listdir(LOG_DIR):
            if filename.endswith('.txt'):
                try:
                    # Extract date from filename (YYYY-MM-DD.txt)
                    file_date_str = filename.replace('.txt', '')
                    file_date = datetime.strptime(file_date_str, DATE_FORMAT).date()
                    
                    if file_date < cutoff_date:
                        file_path = os.path.join(LOG_DIR, filename)
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"🗑️ Deleted old log: {filename}")
                except (ValueError, OSError) as e:
                    logger.info(f"⚠️ Could not delete {filename}: {e}")
        
        # 2. Clean candle data files in DATA_OUTPUT_DIR
        if os.path.exists(DATA_OUTPUT_DIR):
            for filename in os.listdir(DATA_OUTPUT_DIR):
                if filename.startswith('nifty_') and filename.endswith('.csv'):
                    try:
                        # Extract date from filename (nifty_YYYY-MM-DD.csv)
                        file_date_str = filename.replace('nifty_', '').replace('.csv', '')
                        file_date = datetime.strptime(file_date_str, DATE_FORMAT).date()
                        
                        if file_date < cutoff_date:
                            file_path = os.path.join(DATA_OUTPUT_DIR, filename)
                            os.remove(file_path)
                            deleted_count += 1
                            logger.info(f"🗑️ Deleted old candle data: {filename}")
                    except (ValueError, OSError) as e:
                        logger.info(f"⚠️ Could not delete {filename}: {e}")
        
        # 3. Clean 5-min candle CSVs in CANDLES_OUTPUT_PATH
        if CANDLES_OUTPUT_PATH.exists():
            for file_path in CANDLES_OUTPUT_PATH.glob('*_candles.csv'):
                try:
                    filename = file_path.name
                    # Extract date from filename (YYYY-MM-DD_candles.csv)
                    file_date_str = filename.replace('_candles.csv', '')
                    file_date = datetime.strptime(file_date_str, DATE_FORMAT).date()
                    
                    if file_date < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"🗑️ Deleted old candles: {filename}")
                except (ValueError, OSError) as e:
                    logger.info(f"⚠️ Could not delete {filename}: {e}")
        
        # 4. Clean profile files (*.prof, *.profile.txt)
        for ext in ['.prof', '.profile.txt']:
            for filename in os.listdir(LOG_DIR):
                if filename.endswith(ext):
                    try:
                        # Extract date from filename (YYYY-MM-DD.prof or YYYY-MM-DD.profile.txt)
                        file_date_str = filename.replace(ext, '')
                        file_date = datetime.strptime(file_date_str, DATE_FORMAT).date()
                        
                        if file_date < cutoff_date:
                            file_path = os.path.join(LOG_DIR, filename)
                            os.remove(file_path)
                            deleted_count += 1
                            logger.info(f"🗑️ Deleted old profile: {filename}")
                    except (ValueError, OSError) as e:
                        logger.info(f"⚠️ Could not delete {filename}: {e}")
        
        if deleted_count > 0:
            logger.info(f"🧹 Cleanup complete: Deleted {deleted_count} old files (retention: {retention_days} days)")
        else:
            logger.info(f"✅ No files to clean (retention: {retention_days} days)")

        # 5. Clean cached spot 5-min data (keep last 5 days only)
        spot_cutoff = (now_ist - timedelta(days=5)).date()
        spot_deleted = 0
        if os.path.exists(SPOT_DATA_DIR):
            for root, dirs, files in os.walk(SPOT_DATA_DIR):
                for fname in files:
                    if fname.endswith('.csv'):
                        try:
                            file_date_str = fname.replace('.csv', '')
                            file_date = datetime.strptime(file_date_str, DATE_FORMAT).date()
                            if file_date < spot_cutoff:
                                fpath = os.path.join(root, fname)
                                os.remove(fpath)
                                spot_deleted += 1
                                logger.info(f"🗑️ Deleted old spot data: {fpath}")
                        except (ValueError, OSError):
                            pass
        if spot_deleted > 0:
            logger.info(f"🧹 Spot data cleanup: Deleted {spot_deleted} files (retention: 5 days)")

    except Exception as e:
        logger.info(f"❌ Cleanup error: {e}")

def configure_daily_logger(date_str: str):
    global logger
    log_path = os.path.join(LOG_DIR, f"{date_str}.txt")

    # REMOVE ONLY existing FileHandlers (keep StreamHandlers)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    # Add fresh FileHandler for the day
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info(f"🗓️ New log file started for {date_str} at {ist_now().strftime('%H:%M:%S IST')}")

# === TOTP + Auth ===
def get_totp(secret):
    return pyotp.TOTP(secret).now()

def get_access_token():
    MAX_AUTH_RETRIES = 2
    RETRY_DELAY = 25   # seconds between retries (> TOTP 30s window for fresh code)
    TIMEOUT = 30       # request timeout in seconds

    last_error = None
    last_response = None
    attempt_logs = []

    for attempt in range(1, MAX_AUTH_RETRIES + 2):  # 1 initial + retries
        try:
            logger.info(f"🔐 Auth attempt {attempt}/{MAX_AUTH_RETRIES + 1}...")
            resp = requests.post(
                ACCESS_TOKEN_URL,
                headers={"Authorization": AUTHORIZATION},
                data={
                    "password": PASSWORD,
                    "twoFa": get_totp(TOTP_SECRET),
                    "twoFaTyp": "totp"
                },
                timeout=TIMEOUT
            )
            last_response = resp

            if resp.status_code == 200:
                data = resp.json()
                if "access_token" in data:
                    logger.info(f"✅ Auth successful on attempt {attempt}")
                    return data["access_token"]
                else:
                    last_error = f"No 'access_token' in response: {data}"
                    logger.info(f"⚠️ {last_error}")
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text}"
                logger.info(f"❌ Auth failed: {last_error}")

        except requests.exceptions.Timeout as e:
            last_error = f"Request timed out after {TIMEOUT}s: {e}"
            logger.info(f"⏱️ {last_error}")
        except requests.exceptions.RequestException as e:
            last_error = f"Request exception: {e}"
            logger.info(f"❌ {last_error}")
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            logger.info(f"🔥 {last_error}")

        attempt_logs.append(f"Attempt {attempt}: {last_error}")

        # Retry if not last attempt
        if attempt <= MAX_AUTH_RETRIES:
            logger.info(f"⏳ Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    # All retries exhausted - send detailed HTML failure email
    now_str = ist_now().strftime('%Y-%m-%d %H:%M:%S IST')
    _attempts_html = "".join(
        f'<li style="margin:4px 0;font-size:13px;color:#333">{log}</li>'
        for log in attempt_logs
    )
    _body = (
        _html_alert(
            "<strong>🚨 AUTH FAILED — ALL RETRIES EXHAUSTED</strong><br>"
            "Trading session could not start. Manual intervention required.", "red")
        + _html_section("Details")
        + _html_kv_table([
            ("Time", now_str),
            ("Account ID", ACCOUNT_ID),
            ("Account Name", ACCOUNT_NAME),
            ("Total Attempts", str(MAX_AUTH_RETRIES + 1)),
            ("Retry Delay", f"{RETRY_DELAY}s"),
            ("Hostname", platform.node()),
            ("Python", platform.python_version()),
            ("Auth URL", ACCESS_TOKEN_URL),
        ])
        + _html_section("Attempt Log")
        + f'<ol style="padding-left:20px;margin:0">{_attempts_html}</ol>'
        + _html_section("Last Response")
        + _html_kv_table([
            ("Status Code", str(last_response.status_code) if last_response else "N/A"),
            ("Body", f'<code style="font-size:12px;word-break:break-all">'
                     f'{last_response.text[:500] if last_response else "N/A"}</code>'),
        ])
    )
    _email = _html_email_wrap(
        "🚨 AUTH FAILED",
        f"Account: {ACCOUNT_ID} · {ACCOUNT_NAME} · {now_str}",
        _body
    )
    send_email("🚨 AUTH FAILED - Trading Session Not Started", _email, html=True)
    logger.info("📧 Auth failure email sent.")

    raise RuntimeError(f"Authentication failed after {MAX_AUTH_RETRIES + 1} attempts: {last_error}")

def on_connect(stream, event):
    logger.info(f"📡 Connected: {event}")
    if event.get("s") == "connected":
        stream.subscribeL1([NIFTY_SYMBOL])
        stream.subscribeOHLC([NIFTY_SYMBOL], 5)
        stream.subscribeEvents(["orders", "trades"])
        logger.info("📊 Subscribed to Nifty L1, 5-min OHLC stream, and Order Events")

# === Get yesterday's IST range ===
def get_yesterday_ist_range():
    today = ist_now()

    # Go back until we hit the most recent trading day
    offset = 1
    while True:
        potential_day = today - timedelta(days=offset)
        if is_trading_day(potential_day):   # ✅ uses holiday + weekday check
            break
        offset += 1

    from_time = potential_day.replace(hour=9, minute=15, second=0, microsecond=0)
    to_time   = potential_day.replace(hour=15, minute=30, second=0, microsecond=0)

    return int(from_time.timestamp()), int(to_time.timestamp()), potential_day.strftime(DATE_FORMAT)

# === Pivot Logic ===
def compute_pivots(high: float, low: float, close: float, label: str = "") -> dict:
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = high + 3 * (pivot - low)
    s4 = low - 3 * (high - pivot)
    r5 = high + 4 * (pivot - low)
    s5 = low - 4 * (high - pivot)
    r6 = high + 5 * (pivot - low)
    s6 = low - 5 * (high - pivot)
    r7 = high + 6 * (pivot - low)
    s7 = low - 6 * (high - pivot)
    r8 = high + 7 * (pivot - low)
    s8 = low - 7 * (high - pivot)
    r9 = high + 8 * (pivot - low)
    s9 = low - 8 * (high - pivot)
    r10 = high + 9 * (pivot - low)
    s10 = low - 9 * (high - pivot)
    
    PIVOT_LEVELS = {
        "pivot": round(pivot, 2),
        "r1": round(r1, 2), "s1": round(s1, 2),
        "r2": round(r2, 2), "s2": round(s2, 2),
        "r3": round(r3, 2), "s3": round(s3, 2),
        "r4": round(r4, 2), "s4": round(s4, 2),
        "r5": round(r5, 2), "s5": round(s5, 2),
        "r6": round(r6, 2), "s6": round(s6, 2),
        "r7": round(r7, 2), "s7": round(s7, 2),
        "r8": round(r8, 2), "s8": round(s8, 2),
        "r9": round(r9, 2), "s9": round(s9, 2),
        "r10": round(r10, 2), "s10": round(s10, 2),
    }
    lines = []
    logger.info(f"\n📊 Pivot Summary{f' for {label}' if label else ''}")
    lines.append(f"📊 Pivot Summary{f' for {label}' if label else ''}")
    logger.info(f"High: {high} | Low: {low} | Close: {close}")
    lines.append(f"High: {high} | Low: {low} | Close: {close}")
    for level, value in PIVOT_LEVELS.items():
        logger.info(f"{level.upper()}: {value}")
        lines.append(f"{level.upper()}: {value}")
    pivot_text = "\n".join(lines)
    if globals().get("ACCOUNT_INDEX", 1) == 1:
        send_email(f"Pivots for {ist_now().strftime(DATE_FORMAT)}", pivot_text)
    return PIVOT_LEVELS

# === OHLC Fetcher ===
def fetch_nifty_5min_ohlc(access_token):
    start_ts, end_ts, dt_str = get_yesterday_ist_range()
    params = {
        "from": start_ts,
        "to": end_ts,
        "interval": INTERVAL,
        "id": NIFTY_SYMBOL_ID_FOR_OHLC
    }

    headers = {
        "Authorization": f"{AUTHORIZATION}:{access_token}",
        "Accept": "application/json"
    }

    logger.info(f"📡 Fetching 5-min data for {dt_str}...")

    response = _api_session.get(CHART_URL, params=params, headers=headers)
    if response.status_code != 200:
        logger.info(f"❌ Failed to fetch chart data: {response.text}")
        return None, dt_str

    bars = response.json().get("d", {}).get("bars", [])
    if not bars:
        logger.info("❌ No bars found in response.")
        return None, dt_str

    candles = []
    for entry in bars:
        candles.append({
            "time": datetime.fromtimestamp(entry[0] // 1000, tz=IST).strftime("%H:%M"),
            "open": entry[1],
            "high": entry[2],
            "low": entry[3],
            "close": entry[4]
        })
    logger.info(f"✅ Fetched {len(candles)} candles for {dt_str}")
    return candles, dt_str

# === CSV Writer ===
def save_to_csv(candles, dt_str):
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(DATA_OUTPUT_DIR, FILENAME_TEMPLATE.format(date=dt_str))
    df = pd.DataFrame(candles)
    df.to_csv(path, index=False)
    logger.info(f"✅ Saved candle data to {path}")
    return df

# === Spot Data Warmup for Supertrend ===
def fetch_spot_day_data(date_obj, access_token_val):
    """Fetch one day's 5-min spot candles and cache to disk. Returns DataFrame or None."""
    dt_str = date_obj.strftime(DATE_FORMAT)
    year_str = date_obj.strftime("%Y")
    month_str = date_obj.strftime("%m")
    cache_dir = os.path.join(SPOT_DATA_DIR, year_str, month_str)
    cache_path = os.path.join(cache_dir, f"{dt_str}.csv")

    # Use cached file if exists
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            if len(df) > 0:
                logger.info(f"📂 Loaded cached spot data: {cache_path} ({len(df)} candles)")
                return df
        except Exception:
            pass

    # Fetch from API
    from_time = date_obj.replace(hour=9, minute=15, second=0, microsecond=0)
    to_time = date_obj.replace(hour=15, minute=30, second=0, microsecond=0)
    params = {
        "from": int(from_time.timestamp()),
        "to": int(to_time.timestamp()),
        "interval": 5,
        "id": NIFTY_SYMBOL_ID_FOR_OHLC
    }
    headers = {
        "Authorization": f"{AUTHORIZATION}:{access_token_val}",
        "Accept": "application/json"
    }

    try:
        resp = _api_session.get(CHART_URL, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.info(f"❌ Failed to fetch spot data for {dt_str}: {resp.text}")
            return None
        bars = resp.json().get("d", {}).get("bars", [])
        if not bars:
            logger.info(f"⚠️ No bars for {dt_str}")
            return None

        rows = []
        for entry in bars:
            rows.append({
                "datetime_ist": datetime.fromtimestamp(entry[0] // 1000, tz=IST).strftime("%Y-%m-%d %H:%M:%S%z"),
                "open": entry[1],
                "high": entry[2],
                "low": entry[3],
                "close": entry[4]
            })
        df = pd.DataFrame(rows)

        # Cache to disk
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"✅ Fetched & cached {len(df)} candles for {dt_str}")
        return df
    except Exception as e:
        logger.info(f"❌ Error fetching spot data for {dt_str}: {e}")
        return None


def warmup_nifty_st(access_token_val):
    """Load last 3 trading days of 5-min spot data and seed _nifty_st."""
    global _nifty_st
    _nifty_st.reset()

    # Find last 3 trading days
    today = ist_now()
    trading_days = []
    offset = 1
    while len(trading_days) < 3 and offset < 15:
        d = today - timedelta(days=offset)
        if is_trading_day(d):
            trading_days.append(d)
        offset += 1

    trading_days.reverse()  # chronological order

    if not trading_days:
        logger.info("⚠️ No trading days found for ST warmup")
        return False

    # Fetch and combine
    all_dfs = []
    for d in trading_days:
        df = fetch_spot_day_data(d, access_token_val)
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    # Also fetch TODAY's candles (9:15 to now) — not cached since day is in progress
    now = ist_now()
    from_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    to_time = now
    params = {
        "from": int(from_time.timestamp()),
        "to": int(to_time.timestamp()),
        "interval": 5,
        "id": NIFTY_SYMBOL_ID_FOR_OHLC
    }
    headers = {
        "Authorization": f"{AUTHORIZATION}:{access_token_val}",
        "Accept": "application/json"
    }
    try:
        resp = _api_session.get(CHART_URL, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            bars = resp.json().get("d", {}).get("bars", [])
            if bars:
                rows = [{"open": e[1], "high": e[2], "low": e[3], "close": e[4]} for e in bars]
                today_df = pd.DataFrame(rows)
                all_dfs.append(today_df)
                logger.info(f"✅ Fetched {len(today_df)} today's candles for ST warmup (9:15 → now)")
            else:
                logger.info("⚠️ No today's candles available yet for ST warmup")
        else:
            logger.info(f"⚠️ Failed to fetch today's candles: {resp.status_code}")
    except Exception as e:
        logger.info(f"⚠️ Error fetching today's candles: {e}")

    if not all_dfs:
        logger.info("⚠️ No spot data loaded for ST warmup")
        return False

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"📊 ST warmup: {len(combined)} candles from {len(all_dfs)} days")

    try:
        _nifty_st.warm_up(combined)
        direction, st_val = _nifty_st.current()
        trend_str = "BULLISH ↑" if direction == 1 else "BEARISH ↓"
        logger.info(f"✅ Nifty ST warmup complete: {trend_str} | ST={st_val}")
        return True
    except Exception as e:
        logger.info(f"❌ ST warmup failed: {e}")
        return False


def fetch_option_5min_candles(option_symbol, access_token_val):
    """Fetch today's 5-min option candles for ST warmup after entry."""
    global _option_st
    _option_st.reset()

    today = ist_now()
    from_time = today.replace(hour=9, minute=15, second=0, microsecond=0)
    to_time = today  # up to now

    params = {
        "from": int(from_time.timestamp()),
        "to": int(to_time.timestamp()),
        "interval": 5,
        "id": option_symbol
    }
    headers = {
        "Authorization": f"{AUTHORIZATION}:{access_token_val}",
        "Accept": "application/json"
    }

    try:
        resp = _api_session.get(CHART_URL, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.info(f"❌ Failed to fetch option candles for {option_symbol}: {resp.text}")
            return False
        bars = resp.json().get("d", {}).get("bars", [])
        if not bars or len(bars) < 2:
            logger.info(f"⚠️ Not enough option bars for ST warmup ({len(bars)} bars)")
            return False

        rows = []
        for entry in bars:
            rows.append({
                "open": entry[1], "high": entry[2], "low": entry[3], "close": entry[4]
            })
        df = pd.DataFrame(rows)
        _option_st.warm_up(df)
        direction, st_val = _option_st.current()
        if direction is not None:
            trend_str = "BULLISH ↑" if direction == 1 else "BEARISH ↓"
            logger.info(f"✅ Option ST warmup: {len(df)} candles | {trend_str} | ST={st_val}")
        else:
            logger.info(f"⚠️ Option ST not ready yet ({len(df)} candles, need ≥10)")
        return True
    except Exception as e:
        logger.info(f"❌ Option ST warmup failed: {e}")
        return False

def get_5min_block(ts: datetime):
    return ts.replace(second=0, microsecond=0, minute=(ts.minute // 5) * 5)

def update_ohlc(ltp, timestamp):
    block = get_5min_block(timestamp)
    ohlc = ohlc_data[block]
    if ohlc["open"] is None:
        ohlc["open"] = ltp
    ohlc["high"] = max(ohlc["high"], ltp)
    ohlc["low"] = min(ohlc["low"], ltp)
    ohlc["close"] = ltp

def print_candle(block, ohlc, source="L1"):
    t = block.strftime("%H:%M") if hasattr(block, 'strftime') else str(block)
    logger.info(f"\n🕔 [{source}] {t}  |  O: {ohlc['open']}  H: {ohlc['high']}  L: {ohlc['low']}  C: {ohlc['close']}")
    print(f"\n🕔 [{source}] {t}  |  O: {ohlc['open']}  H: {ohlc['high']}  L: {ohlc['low']}  C: {ohlc['close']}")

def save_candle_to_csv(timestamp, ohlc):
    path = CANDLES_OUTPUT_PATH / CANDLES_FILENAME_TEMPLATE.format(date=timestamp.strftime("%Y-%m-%d"))
    os.makedirs(path.parent, exist_ok=True)
    df = pd.DataFrame([{
        "time": timestamp.strftime("%H:%M"),
        "open": ohlc["open"],
        "high": ohlc["high"],
        "low": ohlc["low"],
        "close": ohlc["close"]
    }])
    if not path.exists():
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

def save_stream_candle_to_csv(block_key, o, h, l, c):
    """Save a completed 5-min OHLC stream candle to a separate CSV."""
    today_str = ist_now().strftime("%Y-%m-%d")
    path = CANDLES_OUTPUT_PATH / STREAM_CANDLES_FILENAME_TEMPLATE.format(date=today_str)
    os.makedirs(path.parent, exist_ok=True)
    # block_key is like "2026-03-06 09:15", extract just the time
    time_str = block_key.split(" ")[-1] if " " in block_key else block_key
    df = pd.DataFrame([{"time": time_str, "open": o, "high": h, "low": l, "close": c}])
    if not path.exists():
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

def _fetch_nfo_closed_dates() -> set[str]:
    """
    Calls Upstox holidays API and returns dates (YYYY-MM-DD) where NFO is effectively closed.
    Mirrors your original logic from determine_expiry_date().
    """
    try:
        url = "https://api.upstox.com/v2/market/holidays"
        data = requests.get(url, headers={"Accept": "application/json"}, timeout=20).json()
        closed = {
            h["date"]
            for h in data.get("data", [])
            if h.get("holiday_type") == "TRADING_HOLIDAY" and (
                "NFO" in (h.get("closed_exchanges") or [])
                or any(
                    oe.get("exchange") == "NFO" and (
                        oe.get("start_time") is not None or oe.get("end_time") is not None
                    )
                    for oe in (h.get("open_exchanges") or [])
                )
            )
        }
        return closed
    except Exception as e:
        print(f"⚠️ Failed to fetch holidays: {e}")
        return set()

def _ensure_holiday_cache_for(now_ist: datetime):
    """
    Refresh the closed-dates cache at most once per IST calendar day.
    """
    global _HOLIDAY_CACHE_DATE, _NFO_CLOSED_DATES
    if _HOLIDAY_CACHE_DATE != now_ist.date():
        _NFO_CLOSED_DATES = _fetch_nfo_closed_dates()
        _HOLIDAY_CACHE_DATE = now_ist.date()
        print(f"📅 NFO holiday cache refreshed: {len(_NFO_CLOSED_DATES)} dates")

def determine_expiry_date():
    global EXPIRY_DATE
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    _ensure_holiday_cache_for(now_ist)

    today = now_ist.date()

    # Always compute the *next* Tuesday (not today even if it's Tuesday)
    days_ahead = (1 - today.weekday() + 7) % 7
    if days_ahead == 0:
        days_ahead = 7
    tuesday = today + timedelta(days=days_ahead)

    for back in (0, 1, 2):
        d = tuesday - timedelta(days=back)
        if d.strftime("%Y-%m-%d") not in _NFO_CLOSED_DATES:
            EXPIRY_DATE = d.strftime("%Y-%m-%d")
            return EXPIRY_DATE

    d = tuesday - timedelta(days=3)
    while d.strftime("%Y-%m-%d") in _NFO_CLOSED_DATES:
        d -= timedelta(days=1)

    EXPIRY_DATE = d.strftime("%Y-%m-%d")
    return EXPIRY_DATE

def is_trading_weekday(d: datetime.date) -> bool:
    return d.weekday() < 5  # Mon–Fri

def is_trading_holiday(d: datetime.date) -> bool:
    return d.strftime("%Y-%m-%d") in _NFO_CLOSED_DATES

def is_trading_day(now_ist: datetime) -> bool:
    """
    Trading day = Mon-Fri AND not an NFO holiday.
    Reuses the same cached holiday set as determine_expiry_date().
    """
    _ensure_holiday_cache_for(now_ist)
    return is_trading_weekday(now_ist.date()) and not is_trading_holiday(now_ist.date())

def should_trigger_trade(candle, pivot_levels):
    h = candle["high"]
    l = candle["low"]
    for level in pivot_levels.values():
        if l <= level <= h:
            return True
    return False

def calculate_strike_price(close, direction):
    if direction == "CE":
        return math.floor(close / 50) * 50
    elif direction == "PE":
        return math.ceil(close / 50) * 50

def _fmt(v, width=None, align="<"):
    """Safe formatter: convert None->'-', then optionally pad/align."""
    s = "-" if v is None else str(v)
    if width is None:
        return s
    return f"{s:{align}{width}}"

current_block = get_5min_block(ist_now())
_nifty_block_finalized = None  # track which block was already finalized (prevent double-processing)

def _finalize_nifty_5m_block(stream):
    """Finalize the current Nifty 5-min stream block: save, print, update ST, run trade logic."""
    global ORDER_PLACED, _pending_pivot_trigger, _active_option_symbol
    global _entry_direction, _entry_st_direction, _nifty_block_finalized

    prev_block = _nifty_5m["block"]
    if prev_block is None:
        return
    if _nifty_block_finalized == prev_block:
        return  # already finalized this block

    prev_o = _nifty_5m["open"]
    prev_h = _nifty_5m["high"]
    prev_l = _nifty_5m["low"]
    prev_c = _nifty_5m["close"]

    if prev_o is None:
        return

    _nifty_block_finalized = prev_block

    # Save + print the COMPLETED stream candle
    save_stream_candle_to_csv(prev_block, prev_o, prev_h, prev_l, prev_c)
    print_candle(prev_block, {"open": prev_o, "high": prev_h, "low": prev_l, "close": prev_c}, source="STREAM")

    # Update ST with the COMPLETED 5-min candle
    st_dir, st_val = _nifty_st.update(prev_o, prev_h, prev_l, prev_c)
    trend_str = "BULL ↑" if st_dir == 1 else ("BEAR ↓" if st_dir == -1 else "N/A")
    logger.info(f"📊 Nifty 5min CLOSED | O:{prev_o} H:{prev_h} L:{prev_l} C:{prev_c} | ST: {trend_str} ({st_val}) | block={prev_block}")

    if st_dir is None:
        return  # ST not ready yet

    now = ist_now()
    side = "sell"
    in_trade_window = (
        datetime.strptime("10:00", "%H:%M").time() <= now.time()
        < datetime.strptime("14:59", "%H:%M").time()
    )

    # ── ACT ON PENDING PIVOT TRIGGER ────────────────────────
    if _pending_pivot_trigger is not None and not ORDER_PLACED:
        trigger = _pending_pivot_trigger
        _pending_pivot_trigger = None  # consumed

        direction = "PE" if st_dir == 1 else "CE"
        atm_strike = calculate_strike_price(trigger["close"], direction)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"🎯 ST CONFIRMED ENTRY | Trigger: {trigger['pivot_name']}={trigger['pivot_level']:.2f}")
        logger.info(f"   ST Direction: {trend_str} → SELL {direction} | Strike: {atm_strike}")
        logger.info(f"   Trigger candle: O:{trigger['open']} H:{trigger['high']} L:{trigger['low']} C:{trigger['close']}")
        logger.info(f"{'=' * 60}\n")

        if ORDER_TYPE == "bracket":
            _t_start = time.perf_counter()
            place_bracket_order(direction, atm_strike, EXPIRY_DATE, side)
            _latency_ms = (time.perf_counter() - _t_start) * 1000
            logger.info(f"⏱️ Signal→Order latency: {_latency_ms:.0f} ms")
            if CURRENT_ORDER_ID and check_order_executed(CURRENT_ORDER_ID):
                ORDER_PLACED = True
                _entry_direction = direction
                _entry_st_direction = st_dir

                logger.info("🎯 Order confirmed executed.")
                order_info = get_single_order_details(CURRENT_ORDER_ID)
                if order_info is not None:
                    send_order_email("Placed & EXECUTED Bracket Order", {"orderId": CURRENT_ORDER_ID, "status": "filled", "avgPrice": order_info.get("avgPrice", 0), "stopLoss": order_info.get("stopTrigPrice", 0), "target": order_info.get("targetPrice", 0)}, latency_ms=_latency_ms)
                    stop_trig_price = order_info.get("avgPrice", 0) + 50
                    target_trig_price = order_info.get("avgPrice", 0) - 50
                    logger.info(f"Modifying BO order, stop: {stop_trig_price}, target: {target_trig_price}")
                    time.sleep(2)
                    active_orders = get_orders("active")
                    results = []

                    for order in active_orders:
                        if order.get("product") == "bracket" and order.get("modifiable") is True:
                            resp, leg_type, payload = modify_bracket_order(order, stop_trig_price, target_trig_price)
                            result = {
                                "orderId": payload.get("orderId") if payload else order.get("orderId"),
                                "symId": payload.get("symId") if payload else order.get("symId"),
                                "qty": payload.get("qty") if payload else order.get("qty"),
                                "avgPrice": payload.get("avgPrice") if payload else order.get("avgPrice"),
                                "legType": leg_type,
                                "limitPrice": payload.get("limitPrice") if payload else None,
                                "status": resp.status_code if resp is not None else "ERR",
                                "resp": resp.text if hasattr(resp, "text") else str(resp),
                            }
                            results.append(result)
                        else:
                            results.append({
                                "orderId": order.get("orderId"),
                                "symId": order.get("symId"),
                                "qty": order.get("qty"),
                                "avgPrice": order.get("avgPrice"),
                                "legType": "-",
                                "limitPrice": "-",
                                "status": "N/A",
                                "resp": "Not modifiable or not a bracket order",
                            })
                            logger.info(f"⚠️ Order {order.get('orderId')} is not modifiable or not a bracket order.")
                    _avg = order_info.get("avgPrice", 0)
                    _summary = _html_alert(
                        f"<strong>Main Order Executed</strong><br>"
                        f"Order <strong>#{CURRENT_ORDER_ID}</strong> filled at ₹{_avg}<br>"
                        f"SL → ₹{stop_trig_price} · Target → ₹{target_trig_price}", "blue")
                    _th = 'padding:8px;text-align:left;color:#5f6368;font-weight:600;font-size:12px;border-bottom:1px solid #e0e0e0'
                    _rows = ""
                    for r in results:
                        _leg = r.get('legType', '-')
                        _lb = _html_badge(_leg.upper(), "red" if _leg == "stoploss" else ("blue" if _leg == "target" else "gray"))
                        _st = str(r.get('status', '-'))
                        _sb = _html_badge(f"✓ {_st}" if _st == "200" else f"✗ {_st}", "green" if _st == "200" else "red")
                        _rows += (
                            f'<tr><td style="padding:8px;border-bottom:1px solid #f0f0f0;font-family:monospace;font-size:12px">{r.get("orderId","-")}</td>'
                            f'<td style="padding:8px;border-bottom:1px solid #f0f0f0">{_lb}</td>'
                            f'<td style="padding:8px;border-bottom:1px solid #f0f0f0;text-align:right">{r.get("qty","-")}</td>'
                            f'<td style="padding:8px;border-bottom:1px solid #f0f0f0;text-align:right;font-weight:600">₹{r.get("limitPrice","-")}</td>'
                            f'<td style="padding:8px;border-bottom:1px solid #f0f0f0;text-align:center">{_sb}</td></tr>')
                    _table = (
                        f'{_html_section("Leg Modifications")}'
                        f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse">'
                        f'<thead><tr style="background:#f8f9fa"><th style="{_th}">Order ID</th><th style="{_th}">Leg</th>'
                        f'<th style="{_th};text-align:right">Qty</th><th style="{_th};text-align:right">New Price</th>'
                        f'<th style="{_th};text-align:center">Status</th></tr></thead><tbody>{_rows}</tbody></table>')
                    _mod_html = _html_email_wrap(
                        "📊 Order Modification Report",
                        f"Account: {ACCOUNT_ID} · {ACCOUNT_NAME} · {ist_now().strftime('%Y-%m-%d %H:%M:%S IST')}",
                        _summary + _table)
                    send_email_async("📊 Order Modification Report", _mod_html, html=True)

                    # Subscribe to option OHLC for ST-flip exit monitoring
                    _active_option_symbol = f"OPTIDX_NIFTY_NFO_{EXPIRY_DATE}_{atm_strike}_{direction}"
                    logger.info(f"📊 Subscribing to option OHLC: {_active_option_symbol}")
                    stream.subscribeOHLC([_active_option_symbol], 5)
                    fetch_option_5min_candles(_active_option_symbol, access_token)
                else:
                    logger.info("❌ No active stoploss/target orders found. No order modification was done.")
        else:
            place_normal_order(direction, atm_strike, EXPIRY_DATE, side)
            if CURRENT_ORDER_ID and check_order_executed(CURRENT_ORDER_ID):
                ORDER_PLACED = True
                _entry_direction = direction
                _entry_st_direction = st_dir
                logger.info("🎯 Order confirmed executed.")
                send_order_email("EXECUTED Normal Order", {"orderId": CURRENT_ORDER_ID, "status": "filled"})
                order_info = get_single_order_details(CURRENT_ORDER_ID)
                if order_info is not None:
                    option = order_info.get("symId", "")
                    qty = order_info.get("qty", 0)
                    side = "buy"
                    stop_trig_price = order_info.get("avgPrice", 0) + 50
                    target_trig_price = order_info.get("avgPrice", 0) - 50
                    logger.info(f"Placing OCO order for {option} with qty {qty}, stop: {stop_trig_price}, target: {target_trig_price}")
                    time.sleep(5)
                    place_oco_order(option, qty, side, stop_trig_price, target_trig_price)

                    _active_option_symbol = option
                    stream.subscribeOHLC([_active_option_symbol], 5)
                    fetch_option_5min_candles(_active_option_symbol, access_token)
                else:
                    logger.info("❌ Order details not found. OCO order not placed.")
        return

    # ── CHECK FOR NEW PIVOT TRIGGER (completed candle) ──────
    if not ORDER_PLACED and in_trade_window and _pending_pivot_trigger is None:
        candle = {"open": prev_o, "high": prev_h, "low": prev_l, "close": prev_c}
        if should_trigger_trade(candle, PIVOT_LEVELS):
            triggered_name = None
            triggered_level = None
            for name, level in PIVOT_LEVELS.items():
                if prev_l <= level <= prev_h:
                    triggered_name = name
                    triggered_level = level
                    break

            _pending_pivot_trigger = {
                "open": prev_o, "high": prev_h, "low": prev_l, "close": prev_c,
                "pivot_name": triggered_name,
                "pivot_level": triggered_level,
                "time": prev_block,
            }
            logger.info(f"🔔 PIVOT TRIGGER set: {triggered_name}={triggered_level:.2f} | Waiting for next candle ST confirmation")

    # ── CHECK DUAL ST FLIP EXIT ─────────────────────────────
    if ORDER_PLACED and _entry_direction is not None:
        st_dir_now, st_val_now = _nifty_st.current()
        _check_st_flip_exit(st_dir_now, st_val_now, stream)


def on_tick(stream, data):
    global current_block, ORDER_PLACED
    global _pending_pivot_trigger, _active_option_symbol, _entry_direction, _entry_st_direction
    side = "sell"
    msg_type = data.get("msgType")

    # ════════════════════════════════════════════════════════════════════════
    # L1 HANDLER — LTP tracking + OHLC building (unchanged)
    # ════════════════════════════════════════════════════════════════════════
    if msg_type == "L1" and data.get("symbol") == NIFTY_SYMBOL:
        ltp = data.get("ltp")
        now = ist_now()
        logger.info(f"[{now.strftime('%H:%M:%S')}] 📈 LTP: {ltp}")

        update_ohlc(ltp, now)

        new_block = get_5min_block(now)
        if new_block != current_block:
            ohlc = ohlc_data[current_block]
            if ohlc["open"] is not None:
                print_candle(current_block, ohlc, source="L1")
                save_candle_to_csv(current_block, ohlc)
            current_block = new_block

            # Finalize the stream's pending 5-min block at the exact boundary
            _finalize_nifty_5m_block(stream)
        return

        return

    # ════════════════════════════════════════════════════════════════════════
    # EVENT HANDLER — Order execution streams
    # ════════════════════════════════════════════════════════════════════════
    if msg_type == "EVENTS":
        logger.info(f"🔔 [EVENT STREAM] Raw: {data}")
        # Typical Tradejini raw event data message tracing
        msg = data.get("message", "")
        if "orderId" in msg or "legType" in msg or "status\":\"filled" in msg:
            logger.info(f"✅ [BROKER EVENT] Order Update: {msg}")
        return

    # ════════════════════════════════════════════════════════════════════════
    # OHLC HANDLER — 5-min candle stream (Nifty + Option)
    # ════════════════════════════════════════════════════════════════════════
    if msg_type == "OHLC":
        symbol = data.get("symbol", "")
        o = data.get("open")
        h = data.get("high")
        l = data.get("low")
        c = data.get("close")
        candle_time = data.get("time", "")

        if o is None or h is None or l is None or c is None:
            return

        # ── NIFTY OHLC ──────────────────────────────────────────────────
        if symbol == NIFTY_SYMBOL:
            block_key = _ohlc_time_to_5m_block(candle_time)
            prev_block = _nifty_5m["block"]

            if prev_block is None:
                # Very first update — start a new 5-min block
                _nifty_5m.update({"block": block_key, "open": o, "high": h, "low": l, "close": c})
                logger.info(f"📊 Nifty OHLC (5m start) | O:{o} H:{h} L:{l} C:{c} | block={block_key} | raw={candle_time}")
                return

            if block_key == prev_block:
                # Same 5-min block — aggregate: keep first open, track H/L, update close
                _nifty_5m["high"] = max(_nifty_5m["high"], h)
                _nifty_5m["low"] = min(_nifty_5m["low"], l)
                _nifty_5m["close"] = c
                st_dir, st_val = _nifty_st.current()
                trend_str = "BULL ↑" if st_dir == 1 else ("BEAR ↓" if st_dir == -1 else "N/A")
                logger.info(f"📊 Nifty OHLC (agg) | O:{_nifty_5m['open']} H:{_nifty_5m['high']} L:{_nifty_5m['low']} C:{c} | ST: {trend_str} ({st_val}) | block={block_key}")
                return

            # ── 5-MIN BLOCK CHANGED → start new block ─────────────
            # Finalize previous block (no-op if L1 already did it)
            _finalize_nifty_5m_block(stream)

            # Start new block with current update
            _nifty_5m.update({"block": block_key, "open": o, "high": h, "low": l, "close": c})
            return

        # ── OPTION OHLC ─────────────────────────────────────────────────
        if _active_option_symbol and symbol == _active_option_symbol:
            block_key = _ohlc_time_to_5m_block(candle_time)
            prev_block = _option_5m["block"]

            if prev_block is None:
                _option_5m.update({"block": block_key, "open": o, "high": h, "low": l, "close": c})
                logger.info(f"📊 Option OHLC (5m start) | O:{o} H:{h} L:{l} C:{c} | block={block_key}")
                return

            if block_key == prev_block:
                _option_5m["high"] = max(_option_5m["high"], h)
                _option_5m["low"] = min(_option_5m["low"], l)
                _option_5m["close"] = c
                opt_dir, opt_val = _option_st.current()
                opt_trend = "BULL ↑" if opt_dir == 1 else ("BEAR ↓" if opt_dir == -1 else "N/A")
                logger.info(f"📊 Option OHLC (agg) | O:{_option_5m['open']} H:{_option_5m['high']} L:{_option_5m['low']} C:{c} | ST: {opt_trend} ({opt_val}) | block={block_key}")
                return

            # Previous 5-min block completed
            p_o = _option_5m["open"]
            p_h = _option_5m["high"]
            p_l = _option_5m["low"]
            p_c = _option_5m["close"]
            _option_5m.update({"block": block_key, "open": o, "high": h, "low": l, "close": c})

            opt_dir, opt_val = _option_st.update(p_o, p_h, p_l, p_c)
            opt_trend = "BULL ↑" if opt_dir == 1 else ("BEAR ↓" if opt_dir == -1 else "N/A")
            logger.info(f"📊 Option 5min CLOSED | O:{p_o} H:{p_h} L:{p_l} C:{p_c} | ST: {opt_trend} ({opt_val}) | block={prev_block}")

            if ORDER_PLACED and _entry_direction is not None:
                nifty_dir, nifty_val = _nifty_st.current()
                _check_st_flip_exit(nifty_dir, nifty_val, stream)

            return


def _check_st_flip_exit(nifty_st_dir, nifty_st_val, stream):
    """Check dual ST flip condition and exit if triggered."""
    global ORDER_PLACED, _entry_direction, _entry_st_direction, _active_option_symbol

    if nifty_st_dir is None or _entry_direction is None:
        return

    opt_dir, opt_val = _option_st.current()
    if opt_dir is None:
        return  # option ST not ready yet

    # Determine if Nifty ST has flipped against our position
    st_flipped = False
    if _entry_direction == "PE" and nifty_st_dir == -1:
        # Sold PE (bullish entry) but Nifty ST turned bearish
        st_flipped = True
    elif _entry_direction == "CE" and nifty_st_dir == 1:
        # Sold CE (bearish entry) but Nifty ST turned bullish
        st_flipped = True

    if not st_flipped:
        return

    # Option ST must be bullish (premium rising) for exit confirmation
    if opt_dir != 1:
        logger.info(f"⚠️ Nifty ST flipped but option ST is BEARISH — holding position")
        return

    # ── DUAL ST FLIP CONFIRMED → EXIT ──────────────────────────────
    logger.info(f"\n{'🚨' * 20}")
    logger.info(f"🚨 DUAL ST FLIP EXIT TRIGGERED!")
    logger.info(f"   Nifty ST: {'BULLISH → BEARISH' if _entry_direction == 'PE' else 'BEARISH → BULLISH'}")
    logger.info(f"   Option ST: BULLISH (premium rising)")
    logger.info(f"   Position: SELL {_entry_direction} | Exiting now...")
    logger.info(f"{'🚨' * 20}\n")

    _exit_start = time.perf_counter()

    # Exit all bracket/OCO legs
    try:
        stoploss_orders = get_orders("stoploss")
        if stoploss_orders:
            exit_stoploss_legs(stoploss_orders)
        else:
            logger.info("⚠️ No stoploss orders found to exit")
    except Exception as e:
        logger.info(f"❌ Error exiting stoploss legs: {e}")

    _exit_ms = (time.perf_counter() - _exit_start) * 1000

    # Send ST flip exit email
    send_st_flip_exit_email(nifty_st_dir, nifty_st_val, opt_dir, opt_val, _exit_ms)

    # Unsubscribe from option OHLC
    if _active_option_symbol:
        try:
            stream.unsubscribeOHLC(5)
            logger.info(f"📊 Unsubscribed from option OHLC: {_active_option_symbol}")
        except Exception as e:
            logger.info(f"⚠️ Error unsubscribing option OHLC: {e}")

    # Mark position as closed
    ORDER_PLACED = True  # Keep True to prevent re-entry today
    _entry_direction = None
    _entry_st_direction = None
    _active_option_symbol = None
    _option_st.reset()

def get_lots_by_margin(sym_id, side, order_type, product):
    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    margin_data = {
        "symId": sym_id,
        "qty": LOT_SIZE,
        "side": side,
        "type": order_type,
        "product": product
    }

    try:
        margin_resp = _api_session.post(MARGIN_URL, headers=headers, data=margin_data)
        margin_resp.raise_for_status()
        required_margin = margin_resp.json().get("d", {}).get("requiredMargin", 0)

        logger.info(f"🧮 Required margin per lot: {required_margin}")

        limits_resp = _api_session.get(LIMITS_URL, headers=headers)
        limits_resp.raise_for_status()
        available_margin = limits_resp.json().get("d", {}).get("availMargin", 0)

        logger.info(f"💰 Available margin: {available_margin}")
        logger.info(f"💰 Available margin minus buffer 100000: {available_margin}")

        if required_margin <= 0:
            logger.info("⚠️ Invalid required margin returned.")
            return 0

        if required_margin > available_margin:
            logger.info("⚠️ Not enough margin to place order.")
            return 0

        lots = math.floor((available_margin) / required_margin)
        logger.info(f"📦 You can trade: {lots} lot(s)")

        return lots

    except Exception as e:
        logger.info(f"❌ Error calculating max tradable lots: {e}")
        return 0

def place_bracket_order(option_type, atm_strike, expiry_date, side):
    global CURRENT_ORDER_ID
    option = f"OPTIDX_NIFTY_NFO_{expiry_date}_{atm_strike}_{option_type}"
    lots_multiplier = get_lots_by_margin(option, "sell", "market", "bracket")

    if lots_multiplier == 0:
        logger.info("❌ Not enough margin to place order.")
        return
    
    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    form_data = {
        "symId": option,
        "qty": lots_multiplier * LOT_SIZE,
        "side": side,
        "type": "market",
        "product": "bracket",
        "validity": "day",
        "stopTrigPrice": "50",
        "targetPrice": "50",
        "remarks": "PivST",
    }

    for key, value in form_data.items():
        logger.info(f"{key}: {value}")

    resp = _api_session.post(BRACKET_ORDER_URL, headers=headers, data=form_data)

    if resp.status_code == 200:
        order_id = resp.json().get("d", {}).get("orderId")
        CURRENT_ORDER_ID = order_id
        logger.info(f"✅ Order placed: {option} | Order ID: {order_id}")
        logger.info(f"✅ Order placed: {resp.json()}")
    else:
        logger.info(f"❌ Order placement failed: {resp.text}")
        send_order_email("FAILED to place Bracket Order", {
            "symId": option, "product": "bracket", "type": "market", "side": side,
            "qty": lots_multiplier * LOT_SIZE, "status": f"{resp.status_code}", "remarks": resp.text[:500]
        })

def place_normal_order(option_type, atm_strike, expiry_date, side):
    global CURRENT_ORDER_ID
    option = f"OPTIDX_NIFTY_NFO_{expiry_date}_{atm_strike}_{option_type}"
    lots_multiplier = get_lots_by_margin(option, "sell", "market", "normal")

    if lots_multiplier == 0:
        logger.info("❌ Not enough margin to place order.")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    form_data = {
        "symId": option,
        "qty": lots_multiplier * LOT_SIZE,
        "side": side,
        "type": "market",
        "product": "normal",
        "validity": "day",
        "remarks": "normal"
    }

    for key, value in form_data.items():
        logger.info(f"{key}: {value}")

    resp = _api_session.post(PLACE_ORDER_URL, headers=headers, data=form_data)

    if resp.status_code == 200:
        response_json = resp.json()
        order_id = response_json.get("d", {}).get("orderId")
        CURRENT_ORDER_ID = order_id
        logger.info(f"✅ Order placed: {option} | Order ID: {order_id}")
        send_order_email("PLACED Normal Order", {"orderId": order_id, "symId": option, "product": "normal", "type": "market", "side": side, "qty": lots_multiplier * LOT_SIZE})
    else:
        logger.info(f"❌ Order placement failed: {resp.status_code} {resp.text}")
        send_order_email("FAILED to place Normal Order", {
            "symId": option, "product": "normal", "type": "market", "side": side,
            "qty": lots_multiplier * LOT_SIZE, "status": f"{resp.status_code}", "remarks": resp.text[:500]
        })

def check_order_executed(order_id):
    headers = {"Authorization": f"Bearer {API_KEY}:{access_token}"}
    params = {"orderId": order_id}
    resp = _api_session.get(ORDER_HISTORY_URL, headers=headers, params=params)
    logger.info(f"Checking order execution status for ID: {order_id}")
    status = resp.json().get("d", {}).get("status")
    if status in ("filled", "completed"):
        logger.info(f"✅ Order {order_id} executed successfully.")
        return True
    else:
        logger.info(f"❌ Order {order_id} not executed. Status: {status}")
        return False

def get_single_order_details(order_id):
    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Accept": "application/json"
    }
    try:
        resp = _api_session.get(GET_ORDER_DETAILS_URL, headers=headers)
        resp.raise_for_status()
        orders = resp.json().get("d", [])

        for order in orders:
            if order.get("orderId") == order_id:
                logger.info("📦 Order found:")
                for k, v in order.items():
                    logger.info(f"   {k}: {v}")
                return order

        logger.info(f"⚠️ Order ID {order_id} not found.")
        return None

    except Exception as e:
        logger.info(f"❌ Failed to fetch order details: {e}")
        return None

def place_oco_order(option, qty, side, stop_trig_price, target_trig_price):
    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "symId": option,
        "side": side,
        "stopLossType": "market",
        "stopLossProduct": "normal",
        "stopTrigPrice": stop_trig_price,
        "stopQty": qty,
        "targetType": "market",
        "targetProduct": "normal",
        "targetTrigPrice": target_trig_price,
        "targetQty": qty,
    }

    try:
        resp = _api_session.post(PLACE_OCO_ORDER_URL, headers=headers, data=data)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"\n✅ OCO order placed: {result}")
        send_order_email("PLACED OCO Order", {"symId": option, "qty": qty, "trigPrice": stop_trig_price, "limitPrice": target_trig_price})
        return result
    except Exception as e:
        logger.info(f"❌ Failed to place OCO order: {e}")
        send_order_email("FAILED to place Normal Order", {
            "symId": option, "product": "normal", "side": side, "status": f"{resp.status_code}", "remarks": resp.text[:500]
        })
        return {}

def get_orders(request_type):
    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Accept": "application/json"
    }
    try:
        logger.info("📥 Fetching all orders...")
        resp = _api_session.get(GET_ORDER_DETAILS_URL, headers=headers)
        resp.raise_for_status()
        all_orders = resp.json().get("d", [])
        logger.info(f"📦 Total orders received: {len(all_orders)}")

        if request_type == "stoploss":
            logger.info("🔍 Filtering for stoploss legs...")
            bracket_stoploss_orders = [
                o for o in all_orders
                if o.get("product") == "bracket"
                and o.get("status", "").lower() == "open"
                and o.get("exitable") is True
                and o.get("legType", "").lower() == "stoploss"
            ]

            if not bracket_stoploss_orders:
                logger.info("✅ No stoploss exits needed.")
                return []
            
            logger.info(f"🎯 Stoploss legs eligible for exit: {len(bracket_stoploss_orders)}")
            return bracket_stoploss_orders
        
        elif request_type == "active":
            active_orders = [
                o for o in all_orders
                if o.get("status", "").lower() in ["open", "trigger pending"]
            ]

            if not active_orders:
                logger.info("✅ No active orders found.")
                return []

            logger.info(f"🎯 Active orders: {len(active_orders)}")
            return active_orders
        
    except Exception as e:
        logger.info(f"❌ Failed to fetch orders: {e}")
        return []

def round_to_tick(price: float, tick: float = 0.05) -> float:
    """Round given price to nearest tick (default = 0.05)."""
    if price is None:
        return None
    return round(round(price / tick) * tick, 2)

def modify_bracket_order(order, stop_trig_price, target_trig_price):
    sym_id = order["symId"]
    order_id = order["orderId"]
    leg_type = order.get("legType", "").lower()
    qty = order.get("qty", 0)

    if leg_type == "stoploss":
        new_trig_price = round_to_tick(stop_trig_price)
        new_limit_price = round_to_tick(stop_trig_price + 0.5)
        order_type = "stoplimit"

    elif leg_type == "target":
        new_trig_price = None
        new_limit_price = target_trig_price
        if new_limit_price < 0:
            new_limit_price = .5
        new_limit_price = round_to_tick(new_limit_price)
        order_type = "limit"

    else:
        logger.info(f"⚠️ Unknown or unhandled leg type: {leg_type}")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    payload = {
        "symId": sym_id,
        "orderId": order_id,
        "qty": qty,
        "type": order_type,
        "validity": "day"
    }
    if new_limit_price is not None:
        payload["limitPrice"] = round(new_limit_price, 2)
    if new_trig_price is not None:
        payload["trigPrice"] = round(new_trig_price, 2)

    logger.info(f"\n🔄 Modifying {leg_type.upper()} order:")
    logger.info(f"   🔹 Symbol: {sym_id}")
    logger.info(f"   🔹 Order ID: {order_id}")
    logger.info(f"   🔹 Quantity: {qty}")
    logger.info(f"   🔹 New Limit Price: {payload.get('limitPrice')}")
    logger.info(f"   🔹 New Trigger Price: {payload.get('trigPrice')}")

    try:
        resp = _api_session.put(MODIFY_ORDER_URL, headers=headers, data=payload)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"✅ Order modified successfully: {result}")
        # if resp.status_code == 200:
        # send_order_email("MODIFIED Bracket Order", {"orderId": order_id, "legType": leg_type, "trigPrice": payload.get("trigPrice"), "limitPrice": payload.get("limitPrice"), "status": f"{resp.status_code}"})
        return resp, leg_type, payload

    except Exception as e:
        logger.info(f"❌ Failed to modify order: {e}")
        send_order_email("FAILED to modify Bracket Order", {
            "symId": sym_id, "product": "bracket", "legType": leg_type,
            "qty": qty, "status": f"{resp.status_code}" if 'resp' in locals() else "ERR", "remarks": resp.text[:500] if 'resp' in locals() else str(e)
        })
        return None, leg_type, payload

def exit_stoploss_legs(stoploss_orders):

    headers = {
        "Authorization": f"Bearer {API_KEY}:{access_token}",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    exited_main_ids = set()
    for order in stoploss_orders:
        main_id = order.get("mainLegOrderId")
        sym_id = order["symId"]
        order_id = order["orderId"]
        qty = order.get("qty", 0)
        leg_type = order.get("legType", "").lower()
        if not main_id or main_id in exited_main_ids:
            continue

        payload = {
            "orderId": main_id,
            "product": "bracket"
        }

        logger.info(f"\n🛑 Exiting STOPLOSS via mainLegOrderId:")
        logger.info(f"   🔹 Stoploss ID: {order['orderId']}")
        logger.info(f"   🔹 MainLeg ID:  {main_id}")
        logger.info(f"   🔹 Symbol:      {order['symId']}")
        logger.info(f"   🔹 TrigPrice:   {order.get('trigPrice')} | LimitPrice: {order.get('limitPrice')}")

        try:
            exit_resp = _api_session.post(EXIT_ORDER_URL, headers=headers, data=payload)
            if exit_resp.status_code == 200:
                logger.info(f"   ✅ Exit submitted: {exit_resp.json()}")
                send_order_email("Exited Bracket Order", {"symbol": sym_id, "orderId": order_id, "qty": qty, "legType": leg_type})
                exited_main_ids.add(main_id)
            else:
                logger.info(f"   ❌ Exit failed: {exit_resp.status_code} | {exit_resp.text}")
                send_order_email("FAILED to exit Bracket Order", {
                    "symId": sym_id, "product": "bracket", "orderId": order_id, "legType": leg_type,
                    "qty": qty, "status": f"{exit_resp.status_code}", "remarks": exit_resp.text[:500]
                })
        except Exception as e:
            logger.info(f"   ❌ Exception during exit: {e}")

def signal_handler(sig, frame):
    global RUNNING
    logger.info(f"🔔 Signal {sig} received; continuing to run per 24x7 requirement.")
    RUNNING = True  # never stop; ignore signals except to log

def _sigterm_handler(sig, frame):
    logger.info("🧹 SIGTERM received: shutting down cleanly for systemd...")
    sys.exit(0)

# --- profiling helper (kept, optional) ---
def run_with_cprofile(func, *, date_str: str, log_dir: str, sort_by: str = "cumulative", lines: int = 50):
    os.makedirs(log_dir, exist_ok=True)
    prof_path = os.path.join(log_dir, f"{date_str}.prof")
    txt_path  = os.path.join(log_dir, f"{date_str}.profile.txt")

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        func()
    finally:
        profiler.disable()
        profiler.dump_stats(prof_path)

        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats(sort_by).print_stats(lines)
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(s.getvalue())

        logging.getLogger().info(f"cProfile results saved to: {prof_path} and {txt_path}")

# === Heartbeat & session lifecycle ===
_last_heartbeat_minute = None
_last_email_hour = None
_session_active = False
_stream = None
_auth_failed_for_date = None  # If set to today's date, skip auth attempts for the day

def heartbeat(now_ist):
    global _last_heartbeat_minute
    if now_ist.minute % 5 == 0 and now_ist.second == 0:
        if _last_heartbeat_minute != now_ist.minute:
            _last_heartbeat_minute = now_ist.minute
            logger.info("💓 Heartbeat: program alive")

def start_session():
    """09:00 bootstrap: token, pivots, stream connect."""
    global _session_active, _stream, access_token, PIVOT_LEVELS, _auth_failed_for_date

    # Skip if auth already failed today (wait until next day)
    today = ist_now().date()
    if _auth_failed_for_date == today:
        return  # Already failed today, don't retry

    try:
        access_token = get_access_token()
    except Exception as e:
        logger.info(f"❌ Auth failed permanently for today: {e}")
        _auth_failed_for_date = today  # Mark as failed, skip until tomorrow
        logger.info("🛑 Auth disabled for today. Will retry tomorrow.")
        return

    candles, dt_str = fetch_nifty_5min_ohlc(access_token)
    if not candles:
        logger.info("⚠️ No candles for pivots; will retry next minute.")
        return

    df = save_to_csv(candles, dt_str)
    high, low, close = df["high"].max(), df["low"].min(), df["close"].iloc[-1]
    PIVOT_LEVELS = compute_pivots(high, low, close, label=dt_str)

    # Warmup Nifty Supertrend with last 3 days of spot data
    warmup_nifty_st(access_token)

    try:
        _stream = NxtradStream(API_HOST, stream_cb=on_tick, connect_cb=on_connect)
        _stream.connect(f"{API_KEY}:{access_token}")
    except Exception as e:
        logger.info(f"❌ Stream connect failed; will retry: {e}")
        _stream = None
        return

    _session_active = True
    logger.info("🚀 Trading session STARTED.")

    try:
        determine_expiry_date()
        logger.info(f"📅 Expiry date for today's trades: {determine_expiry_date()}")
    except Exception as e:
        logger.info(f"❌ Determine expiry date failed {e}")

def stop_session():
    """15:30: disconnect & reset."""
    global _session_active, _stream
    try:
        if _stream:
            _stream.disconnect()
    except Exception as e:
        logger.info(f"⚠️ Stream disconnect issue: {e}")
    _stream = None

    reset_state("EOD 15:30")
    _session_active = False
    logger.info("🛑 Trading session STOPPED (outside market hours).")

# === Forever run loop ===
def run_forever():
    """
    Runs 24x7:
      - midnight: new log file + full state reset
      - 09:00-15:30 IST: trading session active
      - outside hours: idle but alive with 5-min heartbeats
      - hourly emails with last-hour logs
    """
    global logger

    today_str = ist_now().strftime(DATE_FORMAT)
    configure_daily_logger(today_str)
    reset_state("startup")

    current_log_date = ist_now().date()

    while True:
        try:
            now = ist_now()

            heartbeat(now)

            if now.hour == 15 and now.minute == 0 and now.second == 0:
                if ORDER_PLACED:
                    logger.info("🔚 3PM: Squaring off open position...")
                    stoploss_orders = get_orders("stoploss")
                    exit_stoploss_legs(stoploss_orders)
                elif not ORDER_PLACED:
                    logger.info("⏱️ 3PM: No trade placed. Good luck for tomorrow!")
            
            # Midnight rollover
            if now.date() != current_log_date:
                current_log_date = now.date()
                today_str = now.strftime(DATE_FORMAT)
                configure_daily_logger(today_str)
                
                # Cleanup old logs (keep last 30 days by default)
                # Adjust retention_days as needed: 7, 14, 30, 60, 90 etc.
                retention_days = int(os.getenv('LOG_RETENTION_DAYS', '7'))
                cleanup_old_logs(retention_days)
                
                # Reset auth failure flag for new day
                global _auth_failed_for_date
                _auth_failed_for_date = None
                logger.info("🔄 Auth failure flag reset for new day.")
                
                reset_state("midnight")

            # Session control
            if is_trading_day(now):
                if in_window(now, ("09","00"), ("15","30")):
                    if not _session_active:
                        start_session()
                else:
                    if _session_active:
                        send_eod_email(now)
                        logger.info("🛌 Trading hours completed. Stopping session.")
                        stop_session()
                    # keep alive; no-op
            else:
                if _session_active:
                    logger.info("🛌 Non-trading day (weekend/holiday). Stopping session.")
                    stop_session()
            time.sleep(1)

        except KeyboardInterrupt:
            logger.info("👋 SIGINT received; ignoring to keep 24x7 loop alive.")
            time.sleep(1)
        except Exception as e:
            logger.exception(f"🔥 Loop error (will keep running): {e}")
            time.sleep(2)  # brief backoff

# === MAIN ===
if __name__ == "__main__":
    # parse CLI and load selected account BEFORE logger path setup
    args = parse_args()
    load_account_env(args.account)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    log_file = os.path.join(LOG_DIR, f"{datetime.now():%Y-%m-%d}.txt")

    logger = logging.getLogger("trader")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(sh)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(sh)

    logger.info(f"✅ Logger initialized — logs will go to file and systemd journal. (account={args.account}, id={ACCOUNT_ID}, name={ACCOUNT_NAME})")

    # Keep your cProfile helper (optional). To enable profiling, set PROFILE=1
    if os.getenv("PROFILE", "0") == "1":
        # create a date_str for profile filenames
        date_str = ist_now().strftime(DATE_FORMAT)
        run_with_cprofile(run_forever, date_str=date_str, log_dir=LOG_DIR)
    else:
        run_forever()
