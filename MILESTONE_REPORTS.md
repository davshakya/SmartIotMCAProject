# Project Milestone Reports

Project Title: Smart Meter IoT Monitoring with Theft and Fault Detection

## 1. Collect Historical Data

In this milestone, historical and baseline smart meter readings were collected and organized for use in model training and system validation. The project uses `sample_data.csv` as the primary source of normal energy-consumption data. In addition to the base dataset, the project includes a simulator that generates continuous meter readings with fields such as device ID, timestamp, voltage, current, and power. This helped establish a representative data foundation for both offline training and live testing. The collected data supports analysis of normal consumption behavior and provides a reference point for detecting suspicious usage patterns.

Deliverables completed:
- baseline dataset prepared in `sample_data.csv`
- live data generation support added through `meter_simulator.py`
- structured smart meter fields standardized for downstream processing

## 2. Preprocess Datasets

In this milestone, the dataset was cleaned and prepared for machine learning. Relevant electrical features were selected, including voltage, current, and power. The normal dataset was labeled appropriately, and additional synthetic samples were generated to represent `THEFT` and `FAULT` conditions so the model could learn multiclass behavior. Theft samples were created using abnormally low power values, while fault samples were created using abnormally high power values. The preprocessing step converted raw readings into a structured training dataset suitable for classification and aligned the training patterns with the simulator behavior used during runtime.

Deliverables completed:
- feature selection finalized as `voltage`, `current`, and `power`
- class labels prepared as `NORMAL`, `THEFT`, and `FAULT`
- synthetic theft and fault samples generated for supervised training
- training-ready dataset pipeline implemented in `train_model.py`

## 3. Implement Algorithms

In this milestone, the core detection and classification algorithms were implemented. The project uses a `RandomForestClassifier` to classify incoming readings into `NORMAL`, `THEFT`, or `FAULT`. The model is trained in `train_model.py` and stored as `model.pkl`. During runtime, `backend.py` loads the trained model, performs inference on each incoming meter record, and returns the predicted class along with a confidence score. The algorithmic workflow supports real-time classification and demonstrates how machine learning can be integrated into a smart-meter monitoring system.

Deliverables completed:
- multiclass ML model implemented using `RandomForestClassifier`
- training script created in `train_model.py`
- inference pipeline implemented in `backend.py`
- confidence scoring added for model predictions

## 4. Deploy Monitoring Agents

In this milestone, the monitoring pipeline was integrated and deployed as a working prototype. The project includes a simulated smart meter agent in `meter_simulator.py`, a streaming layer using Kafka, a Spark Structured Streaming job in `spark_stream.py`, and a FastAPI backend for classification. The launcher script `run_project.py` automates startup of the major services and also supports a direct fallback mode when Spark is unavailable. This milestone demonstrates end-to-end data flow from simulated devices to prediction services, which is essential for real-time IoT monitoring.

Deliverables completed:
- smart meter simulator implemented for multi-device data generation
- Kafka-based message flow configured for streaming mode
- Spark consumer implemented to forward readings to the backend
- automatic startup and fallback logic added in `run_project.py`

## 5. Create Dashboards

In this milestone, a live monitoring dashboard was designed and implemented to visualize incoming smart meter data and prediction results. The dashboard is served by `flask_app.py` and rendered through `templates/index.html`. It displays a live power graph, prediction summaries, recent event cards, simulator labels, detected events, and model confidence values. The interface helps users observe system behavior in real time and makes the project easier to demonstrate, analyze, and explain during presentations and evaluations.

Deliverables completed:
- live dashboard implemented with Flask and Chart.js
- real-time graph added for power monitoring
- event cards added for latest readings and classifications
- counters and confidence display integrated into the UI

## 6. Test and Document

In this milestone, the system was tested in both streaming mode and fallback direct mode to confirm that predictions and dashboard updates work correctly. Functional checks were performed for training, inference, simulator output, and live visualization. The project was also documented thoroughly in `README.md`, which explains architecture, setup, workflow, and limitations. Additional viva preparation and explanation material was prepared in `VIVA.md` to support academic presentation. This milestone ensured that the project is understandable, demonstrable, and ready for submission.

Deliverables completed:
- system behavior verified across training, simulation, prediction, and dashboard flow
- fallback mode included to improve demo reliability
- documentation written in `README.md`
- viva and presentation notes prepared in `VIVA.md`

## Short Status Summary

- Collect Historical Data: Completed
- Preprocess Datasets: Completed
- Implement Algorithms: Completed
- Deploy Monitoring Agents: Completed
- Create Dashboards: Completed
- Test and Document: Completed
