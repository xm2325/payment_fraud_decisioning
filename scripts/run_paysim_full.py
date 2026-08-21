from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_decisioning.paysim_full import run_full_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Full 6.36M-row PaySim benchmark using DuckDB point-in-time SQL")
    ap.add_argument("--parquet-glob", required=True, help="Glob covering all canonical PaySim parquet shards")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "paysim_full")
    args = ap.parse_args()
    summary = run_full_benchmark(args.parquet_glob, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
