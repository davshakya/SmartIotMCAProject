# meter_simulator.py
# Simulates multiple devices with realistic patterns

from kafka import KafkaProducer
import json, random, time
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

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

print("Starting live meter simulator...")

while True:
    for device in DEVICE_IDS:
        data = generate_data(device)
        producer.send("smart-meter", value=data)
        print(data)
    time.sleep(2)
