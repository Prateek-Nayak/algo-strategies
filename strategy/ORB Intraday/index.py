"""
Dashboard wrapper for the ORB Intraday strategy from orb-multibroker.

This legacy wrapper is kept only for compatibility. It must not overwrite the
submodule's repo-local `.env`, because that breaks the standalone runtime.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--account", type=int, required=True)
args = parser.parse_args()
idx = args.account

wrapper_dir = Path(__file__).resolve().parent
strategy_root = wrapper_dir.parent.parent
orb_repo = strategy_root / "orb-multibroker"

if not orb_repo.exists():
    print(f"[ERROR] orb-multibroker repo not found at {orb_repo}")
    sys.exit(1)

source_ip = os.environ.get(f"OUTBOUND_IP_{idx}", "")
client_id = os.environ.get(f"DHAN_CLIENT_ID_{idx}", "")
pin = os.environ.get(f"DHAN_PIN_{idx}", "")
totp_secret = os.environ.get(f"DHAN_TOTP_SECRET_{idx}", "")
log_dir = os.environ.get(f"LOG_DIR_{idx}", "")

if not client_id or not pin or not totp_secret:
    print(f"[ERROR] Missing Dhan credentials for account {idx}")
    print(f"  DHAN_CLIENT_ID_{idx}={'SET' if client_id else 'MISSING'}")
    print(f"  DHAN_PIN_{idx}={'SET' if pin else 'MISSING'}")
    print(f"  DHAN_TOTP_SECRET_{idx}={'SET' if totp_secret else 'MISSING'}")
    sys.exit(1)

proc_env = os.environ.copy()
proc_env["PYTHONIOENCODING"] = "utf-8"
proc_env["DHAN_CLIENT_ID"] = client_id
proc_env["DHAN_PIN"] = pin
proc_env["DHAN_TOTP_SECRET"] = totp_secret
proc_env["DHAN_AUTO_REFRESH"] = "true"

# IP binding: set env var for registry.py to create an IP-bound session.
# All API calls (auth, data, orders) will originate from this IP.
if source_ip:
    proc_env["ALGO_DASHBOARD_SOURCE_IP"] = source_ip

# Shared token file: per-account token file for cross-process token sharing
secrets_dir = orb_repo / ".secrets"
token_file = secrets_dir / f"dhan_token_{client_id}.txt"
proc_env["DHAN_ACCESS_TOKEN_FILE"] = str(token_file)

preamble = """
import sys
sys.argv = ["multi_broker", "--mode", "live", "--strategy-id", "orb_intraday"]
from multi_broker.__main__ import main
main()
"""

src_path = orb_repo / "src"
if src_path.exists():
    proc_env["PYTHONPATH"] = str(src_path) + os.pathsep + proc_env.get("PYTHONPATH", "")

print(f"[WRAPPER] Launching ORB Intraday for account {idx}")
print(f"[WRAPPER] Repo: {orb_repo}")
print(f"[WRAPPER] IP binding: {source_ip or 'disabled (no OUTBOUND_IP)'}")
print(f"[WRAPPER] Token file: {token_file}")
print(f"[WRAPPER] Log dir: {log_dir}")

result = subprocess.run(
    [sys.executable, "-c", preamble],
    cwd=str(orb_repo),
    env=proc_env,
)

sys.exit(result.returncode)

