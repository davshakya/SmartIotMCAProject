# Smart Meter Project Viva Notes

This file contains viva preparation material for the Smart Meter IoT Monitoring with Theft and Fault Detection project.

## College Viva Questions and Answers

### 1. What is the main aim of this project?

This project aims to monitor smart meter readings in real time and classify them as `NORMAL`, `THEFT`, or `FAULT` using a machine learning model. It also visualizes live results on a dashboard.

### 2. Why is this project important?

It is important because smart metering helps detect electricity theft, equipment faults, and unusual energy usage patterns. This can reduce losses and improve power system reliability.

### 3. What technologies are used in this project?

This project uses:

- Python
- FastAPI
- Flask
- Chart.js
- Apache Kafka
- Apache Spark
- scikit-learn
- pandas
- joblib

### 4. What is the role of Kafka in this project?

Kafka is used as the message broker. It receives live smart meter readings from the simulator and passes them into the streaming pipeline.

### 5. What is the role of Spark in this project?

Spark Structured Streaming reads the data from Kafka and forwards it to the FastAPI backend for machine learning prediction.

### 6. What is the role of FastAPI here?

FastAPI is used as the prediction backend. It loads the trained model, receives incoming readings, performs prediction, and returns enriched data for the dashboard.

### 7. What is the role of Flask here?

Flask serves the web dashboard and fetches the latest processed data from the FastAPI backend.

### 8. What machine learning model is used?

The current version uses `RandomForestClassifier`.

### 9. Why did you choose RandomForestClassifier?

It is simple, reliable, works well on structured tabular data, handles nonlinear decision boundaries, and is easy to use for multiclass classification.

### 10. What are the input features for the model?

The input features are:

- `voltage`
- `current`
- `power`

### 11. What are the output classes of the model?

The model predicts:

- `NORMAL`
- `THEFT`
- `FAULT`

### 12. How is the training data prepared?

The project uses `sample_data.csv` as the normal class. Then it creates synthetic theft and fault samples by modifying the power values to match the simulator’s low-power and high-power behavior.

### 13. Is this a supervised or unsupervised model?

The current version is supervised because it is trained with class labels: `NORMAL`, `THEFT`, and `FAULT`.

### 14. What was the previous model and how is the new model different?

The earlier version used `IsolationForest`, which only detected whether a reading was anomalous or normal. The new `RandomForestClassifier` directly predicts one of the three classes.

### 15. How does prediction happen during runtime?

When a new reading arrives:

1. the backend extracts `voltage`, `current`, and `power`
2. the model predicts the class
3. the backend adds `prediction` and `confidence`
4. the dashboard displays the result

### 16. What is the meaning of confidence in this project?

Confidence shows how strongly the classifier supports its predicted class. It is calculated using the model’s class probabilities.

### 17. What is the purpose of the simulator label?

The simulator label acts like expected output or ground truth in the demo. It allows comparison between the actual injected event and the model prediction.

### 18. Why does the project have both a label and a prediction?

The `label` comes from the simulator, while the `prediction` comes from the ML model. Comparing them helps evaluate whether the model is working correctly.

### 19. What happens if Spark fails?

The launcher automatically falls back to direct mode. In that mode, the simulator sends data directly to the backend so the dashboard still works.

### 20. Why did you add fallback mode?

Fallback mode improves reliability during demo and development. Even if Kafka or Spark is unavailable, the project can still run and show predictions on the dashboard.

### 21. What is KRaft mode in Kafka?

KRaft mode is Kafka’s newer mode where Kafka runs without Zookeeper. It simplifies setup and reduces infrastructure complexity.

### 22. What are the limitations of this project?

Main limitations are:

- the theft and fault training data are synthetic
- backend data is stored only in memory
- no database persistence is used
- the system is designed for demo or academic use, not production

### 23. Why are synthetic labels a limitation?

Because the model is not trained on real field data. It learns patterns that match the simulator well, but it may not generalize to real-world smart grid conditions.

### 24. How can this project be improved in the future?

Possible improvements:

- use real labeled smart meter data
- add database storage
- evaluate model accuracy formally
- deploy with Docker
- support user authentication
- add device-wise filtering and reports

### 25. How would you explain the dashboard to an examiner?

The dashboard shows live meter readings, model predictions, simulator labels, confidence scores, and summary counts. It helps visually monitor whether the system is identifying theft and fault events correctly.

### 26. How is theft represented in this project?

Theft is represented mainly by abnormally low power readings, typically near `0.0` to `0.5`.

### 27. How is fault represented in this project?

Fault is represented by abnormally high power readings, typically near `20.0` to `30.0`.

### 28. What is the difference between streaming mode and direct mode?

In streaming mode, data flows through Kafka and Spark before reaching the backend. In direct mode, the simulator sends readings straight to the backend without Kafka or Spark.

### 29. Why did you use both Flask and FastAPI instead of only one framework?

FastAPI is good for fast backend inference APIs, while Flask is simple for serving HTML templates and dashboard pages. Splitting them keeps responsibilities clear.

### 30. If the examiner asks whether this is production-ready, what should you say?

You should say it is a working prototype and academic demo. It clearly demonstrates the architecture and ML workflow, but it still needs real labeled data, persistence, evaluation, and security improvements before production use.

## Quick Viva Revision

### 1-Minute Project Introduction

My project is a Smart Meter IoT Monitoring System with theft and fault detection using machine learning.
It simulates live smart meter readings such as voltage, current, and power.
These readings are processed through a backend where a trained machine learning model classifies them as `NORMAL`, `THEFT`, or `FAULT`.
The system also includes a live dashboard that shows the power timeline, prediction results, and recent events.
I used Python, FastAPI, Flask, Chart.js, Kafka, Spark, and scikit-learn in this project.
The main goal is to demonstrate how real-time smart meter data can be monitored and analyzed for suspicious behavior.

### Teacher-Style Self Introduction

Good morning sir/madam.
My project title is Smart Meter IoT Monitoring with Theft and Fault Detection.
In this project, I worked on simulating real-time meter data, applying machine learning for classification, and building a live dashboard for visualization.
The system predicts whether a reading is normal, theft, or fault based on voltage, current, and power values.
This project helped me understand IoT simulation, streaming concepts, backend APIs, dashboards, and ML-based classification.

### Very Short Explanation of Normal, Theft, and Fault

In this project, the model classifies readings as normal, theft, or fault based on voltage, current, and power.
Normal readings stay in the expected power range.
Theft is identified when power becomes abnormally low.
Fault is identified when power becomes abnormally high.
The classifier learns these patterns from the training data and predicts the class for each live meter reading.

### 10 Short Viva Answers

#### 1. What is the objective of your project?

To detect normal, theft, and fault conditions from smart meter readings in real time.

#### 2. Which ML model did you use?

I used `RandomForestClassifier`.

#### 3. What are the input features?

Voltage, current, and power.

#### 4. What are the output classes?

`NORMAL`, `THEFT`, and `FAULT`.

#### 5. Why did you use FastAPI?

To serve the ML prediction backend efficiently.

#### 6. Why did you use Flask?

To serve the dashboard webpage.

#### 7. What is the role of Kafka?

Kafka is used to stream live smart meter readings.

#### 8. What is the role of Spark?

Spark consumes Kafka data and forwards it for prediction.

#### 9. How is theft detected in your project?

Theft is associated with abnormally low power values.

#### 10. How is fault detected in your project?

Fault is associated with abnormally high power values.
