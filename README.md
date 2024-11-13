# Outliers Detection and Distribution Shift Scoring
This project was created as part of a technical test for Irly consulting, aimed at meeting the requirements for a data scientist position. The code is organized into two branches:

1. model_training Branch: Contains the code for training the outlier detection and distribution shift scoring models, along with an API to serve these models.
2. prod_branch: Features a more scalable architecture with MLflow for model tracking and MinIO to simulate a cloud storage environment.
## Project Context and Description
The objective of this project is to perform anomaly detection and evaluate distribution shifts in Amazon reviews data. The goal is to train models for outlier identification and quantify the deviation of each sample from the training distribution. A real-time API is provided to serve these models.

### Key Tasks:
- Dataset Selection: Working with specific categories from the Amazon reviews dataset.
- Outlier Detection: Implementing anomaly detection techniques, with explanations of method choices and evaluation metrics.
- Distribution Shift Scoring: Calculating scores to assess how much each sample deviates from the training distribution.
- API Service: Exposing the models through a REST API for real-time predictions.
### Architecture and Branches
#### model_training Branch: Models and API
- Anomaly Detection Models: Using Isolation Forest, Local Outlier Factor, and Autoencoders.
- API: Built with FastAPI to expose endpoints for outlier detection and scoring.
- Docker: Containerization of the API and dependencies for easy deployment.
#### prod_branch: MLflow and MinIO Architecture
The advanced branch introduces an integrated, orchestrated architecture:
- Storage: MinIO simulates a cloud storage environment.
- Model Tracking: MLflow for model versioning and tracking.
- Orchestration: Apache Airflow to automate the end-to-end pipeline.

The advanced architecture includes the following services:

- Airflow to manage the end-to-end workflow.
- MinIO for raw data and model artifact storage.
- MLflow for experiment tracking and model management.
- FastAPI to serve models via a REST API.