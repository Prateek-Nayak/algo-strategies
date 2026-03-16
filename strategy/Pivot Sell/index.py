from datetime import datetime, timedelta
import math, os
import random
import sys
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
    global ACCOUNT_ID, ACCOUNT_NAME, API_KEY, PASSWORD, TOTP_SECRET, AUTHORIZATION, LOG_DIR

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

RUNNING = True
ORDER_TYPE = "bracket"  # "normal" or "bracket"
ORDER_PLACED = False
CURRENT_ORDER_ID = None
EXPIRY_DATE = None
ohlc_data = defaultdict(lambda: {
    "open": None, "high": float("-inf"), "low": float("inf"), "close": None
})

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
               attachment_bytes: bytes = None):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM and MAIL_TO):
        logger.info("✉️ Email not configured; skipping send.")
        return False
    
    # --- JITTER: small random delay so 20 instances don't all send at once ---
    try:
        delay = random.uniform(0, 5) 
        logger.info(f"⏱️ Email jitter delay: {delay:.2f}s before sending.")
        time.sleep(delay)
    except Exception:
        # In case random/time have any issue, don't block sending
        pass

    try:
        if not attachment_bytes:
            # Plain email (no attachment)
            msg = MIMEText(body, "plain", "utf-8")
        else:
            # Email with attachment
            msg = MIMEMultipart()
            msg.attach(MIMEText(body, "plain", "utf-8"))

            # If caller passed a file path instead of bytes → handle it
            if isinstance(attachment_bytes, str):
                size = os.path.getsize(attachment_bytes)
                if size > 20 * 1024 * 1024:  # >20 MB
                    body += f"\n\n⚠️ Log file too large ({size//1024//1024} MB). Saved at {attachment_bytes}."
                    msg = MIMEText(body, "plain", "utf-8")
                else:
                    with open(attachment_bytes, "rb") as f:
                        part = MIMEApplication(f.read(), _subtype="gzip")
                    fname = attachment_name or os.path.basename(attachment_bytes)
                    part.add_header("Content-Disposition", "attachment", filename=fname)
                    msg.attach(part)
            else:
                # Normal byte-attachment case
                part = MIMEApplication(attachment_bytes, _subtype="gzip")
                part.add_header("Content-Disposition", "attachment", filename=attachment_name or "attachment.txt.gz")
                msg.attach(part)

        msg["Subject"] = f"{ACCOUNT_ID} : {ACCOUNT_NAME} " + subject
        msg["From"] = MAIL_FROM
        msg["To"] = MAIL_TO
        msg["Date"] = formatdate(localtime=True)
        # --- RETRY LOOP AROUND THE EXISTING SMTP LOGIC ---
        MAX_RETRIES = 3
        BASE_BACKOFF = 30  # seconds

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

                # Temporary SMTP errors (4xx) → retry with backoff
                if 400 <= code < 500 and attempt < MAX_RETRIES:
                    sleep_for = BASE_BACKOFF * attempt  # 30s, 60s, 90s
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

def send_order_email(event: str, info: dict):
    """
    event: 'PLACED' | 'EXECUTED' | 'FAILED' | 'MODIFIED' | 'OCO_PLACED'
    info:  dict with order fields you want to surface
    """
    now = ist_now().strftime('%Y-%m-%d %H:%M:%S IST')
    # Pick only the essentials to keep it short
    fields = [
        "orderId","symId","product","type","side","qty","avgPrice","status",
        "remarks","legType","mainLegOrderId","trigPrice","limitPrice","stopTrigPrice","targetPrice"
    ]
    lines = [f"{k}: {info.get(k)}" for k in fields if k in info]
    body = f"Event: {event}\nTime: {now}\n\n" + "\n".join(lines)
    subject = f"Order {event}: {info.get('orderId') or info.get('symId')}"
    send_email(subject, body)

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
    ORDER_PLACED = False
    CURRENT_ORDER_ID = None
    EXPIRY_DATE = None
    ohlc_data.clear()
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
    resp = requests.post(
        ACCESS_TOKEN_URL,
        headers={"Authorization": AUTHORIZATION},
        data={
            "password": PASSWORD,
            "twoFa": get_totp(TOTP_SECRET),
            "twoFaTyp": "totp"
        }
    )
    resp.raise_for_status()
    logger.info(resp.json())
    return resp.json()["access_token"]

def on_connect(stream, event):
    logger.info(f"📡 Connected: {event}")
    if event.get("s") == "connected":
        stream.subscribeL1([NIFTY_SYMBOL])

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

    response = requests.get(CHART_URL, params=params, headers=headers)
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

def print_candle(block, ohlc):
    t = block.strftime("%H:%M")
    logger.info(f"\n🕔 {t}  |  O: {ohlc['open']}  H: {ohlc['high']}  L: {ohlc['low']}  C: {ohlc['close']}")
    print(f"\n🕔 {t}  |  O: {ohlc['open']}  H: {ohlc['high']}  L: {ohlc['low']}  C: {ohlc['close']}")

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
    if d.strftime("%Y-%m-%d") == "2026-02-01":
        return True  # Special holiday
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

def on_tick(stream, data):
    global current_block, ORDER_PLACED
    side = "sell"

    if data.get("msgType") == "L1" and data.get("symbol") == NIFTY_SYMBOL:
        ltp = data.get("ltp")
        now = ist_now()
        logger.info(f"[{now.strftime('%H:%M:%S')}] 📈 LTP: {ltp}")

        update_ohlc(ltp, now)

        new_block = get_5min_block(now)
        if new_block != current_block:
            ohlc = ohlc_data[current_block]
            if ohlc["open"] is not None:
                print_candle(current_block, ohlc)
                save_candle_to_csv(current_block, ohlc)
                if (
                    not ORDER_PLACED and
                    datetime.strptime("10:00", "%H:%M").time() <= current_block.time() < datetime.strptime("14:59", "%H:%M").time()
                    and should_trigger_trade(ohlc, PIVOT_LEVELS)
                ):
                    direction = "PE" if ohlc["close"] > ohlc["open"] else "CE"
                    atm_strike = calculate_strike_price(ohlc["close"], direction)
                    if ORDER_TYPE == "bracket":
                        place_bracket_order(direction, atm_strike, EXPIRY_DATE, side)
                        if CURRENT_ORDER_ID and check_order_executed(CURRENT_ORDER_ID):
                            ORDER_PLACED = True
                            
                            logger.info("🎯 Order confirmed executed.")
                            order_info = get_single_order_details(CURRENT_ORDER_ID)
                            if order_info is not None:
                                send_order_email("Placed & EXECUTED Bracket Order", {"orderId": CURRENT_ORDER_ID, "status": "filled", "avgPrice": order_info.get("avgPrice", 0), "stopLoss": order_info.get("stopTrigPrice", 0), "target": order_info.get("targetPrice", 0)})
                                stop_trig_price = order_info.get("avgPrice", 0) + 50
                                target_trig_price = order_info.get("avgPrice", 0) - 50
                                logger.info(f"Modifying BO order, stop: {stop_trig_price}, target: {target_trig_price}")
                                time.sleep(2)
                                active_orders = get_orders("active")
                                results = []  # collect per-order outcomes

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
                                            "legType": "-",                # not modified
                                            "limitPrice": "-",             # no change
                                            "status": "N/A",
                                            "resp": "Not modifiable or not a bracket order",
                                        })
                                        logger.info(f"⚠️ Order {order.get('orderId')} is not modifiable or not a bracket order.")
                                header = (
                                    f"{_fmt('Order ID',15)}"
                                    f"{_fmt('Sym ID',12)}"
                                    f"{_fmt('Qty',6)}"
                                    f"{_fmt('Avg Price',12)}"
                                    f"{_fmt('Leg Type',10)}"
                                    f"{_fmt('New Limit Price',16)}"
                                    f"{_fmt('Status',8)}"
                                    f"Response"
                                )

                                lines = [header, "-" * 140]

                                for r in results:
                                    line = (
                                        f"{_fmt(r.get('orderId'),15)}"
                                        f"{_fmt(r.get('symId'),12)}"
                                        f"{_fmt(r.get('qty'),6)}"
                                        f"{_fmt(r.get('avgPrice'),12)}"
                                        f"{_fmt(r.get('legType'),10)}"
                                        f"{_fmt(r.get('limitPrice'),16)}"
                                        f"{_fmt(r.get('status'),8)}"
                                        f"{_fmt(r.get('resp'))}"
                                    )
                                    lines.append(line)

                                email_body = "\n".join(lines)
                                send_email(f"📊 {ACCOUNT_ID} : {ACCOUNT_NAME} Order Modification Report", email_body)
                            else:
                                logger.info("❌ No active stoploss/target orders found. No order modification was done.")
                    else:
                        place_normal_order(direction, atm_strike, EXPIRY_DATE, side)
                        if CURRENT_ORDER_ID and check_order_executed(CURRENT_ORDER_ID):
                            ORDER_PLACED = True
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
                                time.sleep(5)  # Wait for order to settle
                                place_oco_order(option, qty, side, stop_trig_price, target_trig_price)
                            else:
                                logger.info("❌ Order details not found. OCO order not placed.")

            current_block = new_block

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
        margin_resp = requests.post(MARGIN_URL, headers=headers, data=margin_data)
        margin_resp.raise_for_status()
        required_margin = margin_resp.json().get("d", {}).get("requiredMargin", 0)

        logger.info(f"🧮 Required margin per lot: {required_margin}")

        limits_resp = requests.get(LIMITS_URL, headers=headers)
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
        "remarks": "PivSell",
    }

    for key, value in form_data.items():
        logger.info(f"{key}: {value}")

    resp = requests.post(BRACKET_ORDER_URL, headers=headers, data=form_data)

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

    resp = requests.post(PLACE_ORDER_URL, headers=headers, data=form_data)

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
    resp = requests.get(ORDER_HISTORY_URL, headers=headers, params=params)
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
        resp = requests.get(GET_ORDER_DETAILS_URL, headers=headers)
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
        resp = requests.post(PLACE_OCO_ORDER_URL, headers=headers, data=data)
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
        resp = requests.get(GET_ORDER_DETAILS_URL, headers=headers)
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
        resp = requests.put(MODIFY_ORDER_URL, headers=headers, data=payload)
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
            exit_resp = requests.post(EXIT_ORDER_URL, headers=headers, data=payload)
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

def heartbeat(now_ist):
    global _last_heartbeat_minute
    if now_ist.minute % 5 == 0 and now_ist.second == 0:
        if _last_heartbeat_minute != now_ist.minute:
            _last_heartbeat_minute = now_ist.minute
            logger.info("💓 Heartbeat: program alive")

def start_session():
    """09:00 bootstrap: token, pivots, stream connect."""
    global _session_active, _stream, access_token, PIVOT_LEVELS

    try:
        access_token = get_access_token()
    except Exception as e:
        logger.info(f"❌ Auth failed; will retry next loop: {e}")
        return

    candles, dt_str = fetch_nifty_5min_ohlc(access_token)
    if not candles:
        logger.info("⚠️ No candles for pivots; will retry next minute.")
        return

    df = save_to_csv(candles, dt_str)
    high, low, close = df["high"].max(), df["low"].min(), df["close"].iloc[-1]
    PIVOT_LEVELS = compute_pivots(high, low, close, label=dt_str)

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