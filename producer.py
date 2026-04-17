from kafka import KafkaProducer
import json, random, time
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_data():
    voltage = random.uniform(210, 240)
    current = random.uniform(5, 15)
    power = voltage * current / 1000

    label = "NORMAL"
    r = random.random()

    if r < 0.05:
        power = random.uniform(0, 0.5)
        label = "THEFT"
    elif r < 0.10:
        power = random.uniform(20, 30)
        label = "FAULT"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "power": round(power, 2),
        "label": label
    }

while True:
    producer.send("smart-meter", value=generate_data())
    time.sleep(1)
