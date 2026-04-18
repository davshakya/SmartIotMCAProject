from __future__ import annotations

from flask import Flask, jsonify, render_template
import requests


BACKEND_URL = "http://localhost:8000/data"

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def data():
    try:
        response = requests.get(BACKEND_URL, timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException:
        return jsonify([])


if __name__ == "__main__":
    app.run(port=5000, debug=True)
