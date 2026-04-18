from __future__ import annotations

import argparse
import random
import time
from datetime import datetime

import requests


DEVICE_IDS = ["METER_1", "METER_2", "METER_3"]


def generate_data(device_id: str) -> dict:
    voltage = random.uniform(210, 240)
    current = random.uniform(5, 15)
    base_power = voltage * current / 1000

    hour = datetime.now().hour
    if 18 <= hour <= 23:
        base_power *= 1.5
    elif 0 <= hour <= 6:
        base_power *= 0.55

    label = "NORMAL"
    random_draw = random.random()
    if random_draw < 0.03:
        base_power = random.uniform(0.0, 0.35)
        label = "THEFT"
    elif random_draw < 0.06:
        base_power = random.uniform(20.0, 30.0)
        label = "FAULT"

    return {
        "device_id": device_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "power": round(base_power, 2),
        "label": label,
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart meter live simulator")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000/data",
        help="Backend endpoint that receives simulated readings.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between each batch of device readings.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    session = requests.Session()
    print("Starting live meter simulator in direct API mode...", flush=True)

    while True:
        for device_id in DEVICE_IDS:
            payload = generate_data(device_id)
            try:
                session.post(args.backend_url, json=payload, timeout=5)
                print(payload, flush=True)
            except requests.RequestException as exc:
                print(f"Failed to send reading to backend: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
