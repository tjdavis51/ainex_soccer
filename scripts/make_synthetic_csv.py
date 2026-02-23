# scripts/make_synthetic_csv.py
from __future__ import annotations

from pathlib import Path
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--servo", type=int, required=True, help="1..22")
    parser.add_argument("--value", type=int, default=None, help="0..1000 (default: 500+delta)")
    parser.add_argument("--delta", type=int, default=200, help="used if --value not provided")
    parser.add_argument("--hold", type=int, default=1000, help="Time in ms for the single frame")
    parser.add_argument("--out", type=str, default=None, help="output csv path (optional)")
    args = parser.parse_args()

    if not (1 <= args.servo <= 22):
        raise SystemExit("servo must be in 1..22")

    if args.value is None:
        value = 500 + args.delta
    else:
        value = args.value

    value = max(0, min(1000, int(value)))

    # Build one frame: all neutral except the selected servo
    servos = [500] * 22
    servos[args.servo - 1] = value

    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "assets" / "action_groups" / "csv"

    if args.out is None:
        fname = f"synth_servo{args.servo:02d}_{value:04d}.csv"
        out_path = default_dir / fname
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Index", "Time"] + [f"Servo{i}" for i in range(1, 23)]
    row = {"Index": 1, "Time": args.hold}
    for i in range(1, 23):
        row[f"Servo{i}"] = servos[i - 1]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"Wrote: {out_path}")
    print(f"Servo{args.servo} set to {value}, others at 500, hold={args.hold} ms")


if __name__ == "__main__":
    main()