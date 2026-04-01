#!/usr/bin/env python
"""Download GLEAM v4.2a monthly actual‑ET (E) via SFTP.

Downloads per‑year netCDFs for 2002–2024 (GRACE overlap period) into
``data/gleam/``.  Skips files that already exist locally.

Usage
-----
    python scripts/download_gleam.py                  # download 2002–2024
    python scripts/download_gleam.py --start 1980     # download from 1980
    python scripts/download_gleam.py --list           # just list remote files
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import paramiko

# ── SFTP credentials (shared GLEAM data‑access account) ──
HOST = "hydras.ugent.be"
PORT = 2225
USER = "gleamuser"
PASS = os.environ.get("GLEAM_PASSWORD", "GLEAM4#h-cel_924")

REMOTE_DIR = "/data/v4.2a/monthly/E"

# ── Local destination ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "gleam"


def connect() -> paramiko.SFTPClient:
    transport = paramiko.Transport((HOST, PORT))
    transport.connect(username=USER, password=PASS)
    return paramiko.SFTPClient.from_transport(transport)


def download(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> None:
    remote_size = sftp.stat(remote_path).st_size or 0
    mb = remote_size / 1e6
    print(f"  {remote_path.rsplit('/', 1)[-1]}  ({mb:.0f} MB)")

    local_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(transferred: int, total: int) -> None:
        pct = transferred / total * 100 if total else 0
        print(f"\r    {transferred / 1e6:7.1f} / {total / 1e6:.0f} MB  ({pct:5.1f}%)",
              end="", flush=True)

    sftp.get(remote_path, str(local_path), callback=progress)
    print(f"\r    Done — {local_path.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GLEAM v4.2a monthly E via SFTP")
    parser.add_argument("--list", action="store_true", help="Just list remote files")
    parser.add_argument("--start", type=int, default=2002, help="First year (default: 2002)")
    parser.add_argument("--end", type=int, default=2024, help="Last year (default: 2024)")
    args = parser.parse_args()

    print(f"Connecting to {HOST}:{PORT} …")
    sftp = connect()
    print("Connected.\n")

    remote_files = {a.filename: a for a in sftp.listdir_attr(REMOTE_DIR)}

    if args.list:
        for name in sorted(remote_files):
            sz = remote_files[name].st_size / 1e6 if remote_files[name].st_size else 0
            print(f"  {name}  ({sz:.0f} MB)")
        sftp.close()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded, skipped = 0, 0

    for year in range(args.start, args.end + 1):
        fname = f"E_{year}_GLEAM_v4.2a_MO.nc"
        if fname not in remote_files:
            print(f"  {fname}  — not on server, skipping")
            continue
        local = OUT_DIR / fname
        if local.exists():
            print(f"  {fname}  — already downloaded")
            skipped += 1
            continue
        download(sftp, f"{REMOTE_DIR}/{fname}", local)
        downloaded += 1

    sftp.close()
    print(f"\nDone: {downloaded} downloaded, {skipped} already existed  →  {OUT_DIR}")


if __name__ == "__main__":
    main()
