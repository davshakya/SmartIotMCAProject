import argparse
import json
import random
import time
from datetime import datetime

import requests

DEVICE_IDS = ["METER_1", "METER_2", "METER_3"]

def generate_data(device_id):
    voltage = random.uniform(210, 240)
    current = random.uniform(5, 15)

    # realistic pattern
    base_power = voltage * current / 1000

    # simulate daily usage variation
    hour = datetime.now().hour
    if 18 <= hour <= 23:
        base_power *= 1.5  # peak usage
    elif 0 <= hour <= 6:
        base_power *= 0.5  # low usage

    label = "NORMAL"

    r = random.random()
    if r < 0.03:
        base_power = random.uniform(0, 0.3)
        label = "THEFT"
    elif r < 0.06:
        base_power = random.uniform(20, 30)
        label = "FAULT"

    return {
        "device_id": device_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "power": round(base_power, 2),
        "label": label
    }


def create_parser():
    parser = argparse.ArgumentParser(description="Smart meter live simulator")
    parser.add_argument(
        "--mode",
        choices=["kafka", "direct"],
        default="kafka",
        help="Send data to Kafka or directly to the FastAPI backend.",
    )
    parser.add_argument(
        "--bootstrap-server",
        default="localhost:9092",
        help="Kafka bootstrap server for kafka mode.",
    )
    parser.add_argument(
        "--topic",
        default="smart-meter",
        help="Kafka topic for kafka mode.",
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000/data",
        help="Backend endpoint for direct mode.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between each batch of simulated device readings.",
    )
    return parser


def run_kafka_mode(args):
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print("Starting live meter simulator in Kafka mode...", flush=True)

    while True:
        for device in DEVICE_IDS:
            data = generate_data(device)
            producer.send(args.topic, value=data)
            print(data, flush=True)
        producer.flush()
        time.sleep(args.interval)


def run_direct_mode(args):
    print("Starting live meter simulator in direct API mode...", flush=True)

    while True:
        for device in DEVICE_IDS:
            data = generate_data(device)
            try:
                requests.post(args.backend_url, json=data, timeout=5)
                print(data, flush=True)
            except requests.RequestException as exc:
                print(f"Failed to send reading to backend: {exc}", flush=True)
        time.sleep(args.interval)


def main():
    args = create_parser().parse_args()
    if args.mode == "direct":
        run_direct_mode(args)
        return
    run_kafka_mode(args)


if __name__ == "__main__":
    main()
