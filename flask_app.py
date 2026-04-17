from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    try:
        res = requests.get("http://localhost:8000/data")
        return jsonify(res.json())
    except:
        return jsonify([])

if __name__ == "__main__":
    app.run(port=5000, debug=True)
