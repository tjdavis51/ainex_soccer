#!/usr/bin/env python3
"""
Extract AINex action group SQLite (.d6a) files into CSV.

Expected layout (repo root):
  assets/action_groups/raw/*.d6a
  assets/action_groups/csv/

Output:
  assets/action_groups/csv/<action_name>.csv

The .d6a files are SQLite databases with a table like:
  ActionGroup(Index INTEGER PRIMARY KEY AUTOINCREMENT, Time INT, Servo1 INT, ... Servo22 INT)
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


TABLE_NAME = "ActionGroup"
SERVO_COUNT = 22


def repo_root() -> Path:
    # tools/ -> repo root
    return Path(__file__).resolve().parents[1]


def expected_columns() -> list[str]:
    cols = ["Index", "Time"]
    cols += [f"Servo{i}" for i in range(1, SERVO_COUNT + 1)]
    return cols


def is_sqlite_file(path: Path) -> bool:
    """
    Quick signature check: SQLite files begin with b"SQLite format 3\\0".
    """
    try:
        with path.open("rb") as f:
            header = f.read(16)
        return header.startswith(b"SQLite format 3")
    except OSError:
        return False


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    )
    return cur.fetchone() is not None


def export_one(db_path: Path, out_csv: Path, overwrite: bool) -> bool:
    """
    Export a single .d6a SQLite DB to CSV.
    Returns True if exported, False if skipped/failed.
    """
    if out_csv.exists() and not overwrite:
        print(f"SKIP  {db_path.name} -> {out_csv.name} (already exists)")
        return False

    if not is_sqlite_file(db_path):
        print(f"SKIP  {db_path.name} (not a SQLite file)")
        return False

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as e:
        print(f"FAIL  {db_path.name} (sqlite open error: {e})")
        return False

    try:
        if not table_exists(conn, TABLE_NAME):
            print(f"SKIP  {db_path.name} (missing table '{TABLE_NAME}')")
            return False

        cols = expected_columns()
        col_list = ", ".join([f'"{c}"' for c in cols])  # quote Index safely
        query = f"SELECT {col_list} FROM {TABLE_NAME} ORDER BY \"Index\" ASC;"

        cur = conn.execute(query)
        rows = cur.fetchall()

        if not rows:
            print(f"SKIP  {db_path.name} (no rows in {TABLE_NAME})")
            return False

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)

        print(f"DONE  {db_path.name} -> {out_csv.name} ({len(rows)} frames)")
        return True

    except sqlite3.Error as e:
        print(f"FAIL  {db_path.name} (sqlite query error: {e})")
        return False
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Export .d6a action groups to CSV")
    ap.add_argument(
        "--raw-dir",
        type=str,
        default=str(repo_root() / "assets" / "action_groups" / "raw"),
        help="Directory containing .d6a files",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(repo_root() / "assets" / "action_groups" / "csv"),
        help="Directory to write CSV files",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise SystemExit(f"Raw directory not found: {raw_dir}")

    d6as = sorted(raw_dir.glob("*.d6a"))
    if not d6as:
        raise SystemExit(f"No .d6a files found in: {raw_dir}")

    exported = 0
    for db_path in d6as:
        out_csv = out_dir / f"{db_path.stem}.csv"
        if export_one(db_path, out_csv, overwrite=args.overwrite):
            exported += 1

    print(f"\nExport complete. Wrote {exported} CSV file(s) to {out_dir}")


if __name__ == "__main__":
    main()