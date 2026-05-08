#!/usr/bin/env python3
"""Fetch mle-bench leaderboard.csv files via GitHub's media CDN.

mle-bench ships per-competition `leaderboard.csv` files via Git LFS. When
the user's git/LFS install can't pull them (corporate proxy, no LFS quota,
managed accounts blocked from LFS), the local files remain ~130-byte
pointer stubs and grading fails with::

    AssertionError: Leaderboard must have a `score` column.

This script downloads the real CSVs from
``https://media.githubusercontent.com/media/openai/mle-bench/main/...``
(no auth, no Git-LFS dependency).

Usage::

    python fetch_leaderboards.py                       # fetch everything missing
    python fetch_leaderboards.py --tasks slug1,slug2   # only specific slugs
    python fetch_leaderboards.py --force               # re-download everything
"""
from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import mlebench

URL_TMPL = (
    "https://media.githubusercontent.com/media/openai/mle-bench/main/"
    "mlebench/competitions/{slug}/leaderboard.csv"
)
COMPS_DIR = Path(mlebench.__file__).parent / "competitions"
MIN_VALID_BYTES = 500
# urllib uses the system OpenSSL trust store by default, which on macOS does
# not include corporate proxy CAs (e.g. Zscaler) that may have been appended
# to certifi's bundle. Build an SSL context from certifi so it also picks up
# anything the user added there.
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def needs_fetch(slug: str) -> tuple[bool, int]:
    p = COMPS_DIR / slug / "leaderboard.csv"
    if not p.exists():
        return True, 0
    sz = p.stat().st_size
    return sz < MIN_VALID_BYTES, sz


def ensure_leaderboard(slug: str) -> None:
    """If the leaderboard CSV for *slug* is missing or is an LFS stub, fetch it.

    Call this before ``mlebench.grade.grade_csv`` so grading never fails
    just because Git LFS was unavailable at install time.
    """
    need, _ = needs_fetch(slug)
    if need:
        ok, msg = fetch_one(slug)
        if not ok:
            raise RuntimeError(f"Cannot fetch leaderboard for {slug}: {msg}")


def fetch_one(slug: str) -> tuple[bool, str]:
    out = COMPS_DIR / slug / "leaderboard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    url = URL_TMPL.format(slug=slug)
    try:
        with urllib.request.urlopen(url, timeout=30, context=_SSL_CTX) as r:
            data = r.read()
    except Exception as e:
        return False, f"download failed: {e}"
    if len(data) < MIN_VALID_BYTES:
        return False, f"got only {len(data)} bytes (likely 404 or pointer)"
    out.write_bytes(data)
    return True, f"{len(data):,} bytes"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", help="Comma-separated slugs (default: every "
                                    "competition mlebench knows about)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if a real CSV already exists.")
    args = ap.parse_args()

    if args.tasks:
        slugs = [s.strip() for s in args.tasks.split(",") if s.strip()]
    else:
        slugs = sorted(
            p.name for p in COMPS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
        )

    print(f"Competitions root: {COMPS_DIR}")
    print(f"Total slugs: {len(slugs)}")

    n_ok = n_skip = n_fail = 0
    failures: list[tuple[str, str]] = []
    for s in slugs:
        need, sz = needs_fetch(s)
        if not need and not args.force:
            n_skip += 1
            continue
        ok, msg = fetch_one(s)
        if ok:
            n_ok += 1
            print(f"  OK   {s}  ({msg})")
        else:
            n_fail += 1
            failures.append((s, msg))
            print(f"  FAIL {s}  {msg}", file=sys.stderr)

    print()
    print(f"Done. fetched={n_ok}  skipped={n_skip}  failed={n_fail}")
    if failures:
        print("Failures:")
        for s, m in failures:
            print(f"  {s}  {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
