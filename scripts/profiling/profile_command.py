#!/usr/bin/env python3
"""Wrap a command with wall-clock timing and metadata recording.

Usage:
  python scripts/profiling/profile_command.py \
    --stage train_round001_seed0 \
    --output experiments/profiling/v100/train_round001_seed0.json \
    -- echo "hello"

Output JSON fields: stage, command, hostname, cwd, git_commit,
  start_time, end_time, elapsed_seconds, exit_code.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a shell command (wall-clock timing + metadata)."
    )
    parser.add_argument("--stage", required=True,
                        help="Label for this profiling stage (e.g. train_round001_seed0).")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path.")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to run (use -- before the command).")
    args = parser.parse_args()

    if not args.command:
        print("ERROR: No command provided. Use -- before the command.", file=sys.stderr)
        print("Example: python scripts/profiling/profile_command.py --stage test --output out.json -- echo hello", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    start_dt = datetime.now(timezone.utc).isoformat()

    result = subprocess.run(args.command, cwd=str(PROJECT_ROOT))
    exit_code = result.returncode

    end_ts = time.time()
    end_dt = datetime.now(timezone.utc).isoformat()
    elapsed = end_ts - start_ts

    record = {
        "stage": args.stage,
        "command": " ".join(args.command),
        "hostname": socket.gethostname(),
        "cwd": str(PROJECT_ROOT),
        "git_commit": _git_commit(),
        "start_time": start_dt,
        "end_time": end_dt,
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": exit_code,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"[profile] stage={args.stage}  elapsed={elapsed:.1f}s  exit={exit_code}")
    print(f"[profile] wrote: {output_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
