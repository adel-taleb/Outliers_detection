# Outliers Detection and Distribution Shift Scoring - Model Training Branch

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Torch-green" alt="Torch">
  <img src="https://img.shields.io/badge/PyOD-Outlier%20Detection-yellow" alt="PyOD">
</p>

This project was developed as part of a technical test for Irly consulting, in the context of applying for a Senior Machine Learning Engineer position. This branch, model_training, focuses on training models for outlier detection and distribution shift scoring, as well as serving these models through an API.

## Project Overview
The objective of this project is to identify anomalies and assess distribution shifts in [Amazon Reviews data 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). This branch provides the code for training models and setting up a real-time API for model predictions.

## Key Components
Outlier Detection: Implements different models to identify outliers in the dataset.
Distribution Shift Scoring: Provides a scoring mechanism to measure deviations from the training distribution.
API: Built with FastAPI to serve the models for real-time predictions.
Docker: The API is containerized for easier deployment and reproducibility.

## Repository Structure
The updated structure of this branch includes:

fast_api/: Contains the FastAPI application code for serving outlier detection and distribution shift scoring.
.gitignore: Specifies files and folders to be ignored in version control.
README.md: This file, providing an overview and instructions for the project.
outliers_detection.py: Main script implementing outlier detection algorithms.
outliers_tester.py: Script for testing and validating the trained models.
requirements.txt: List of dependencies required to run the project.
train.ipynb: Jupyter notebook for model training and experimentation.
tester.ipynb: Jupyter notebook for testing the outlier detection models.

## Setup Instructions
1. **Clone the Repository:**

```
git clone https://github.com/adel-taleb/Outliers_detection.git
cd Outliers_detection
git checkout model_training
```
2. **Install Dependencies:** Ensure you have Python 3.8 or later. Install dependencies from requirements.txt:

```
pip install -r requirements.txt
```
3. **Run the FastAPI Service:** Navigate to the fast_api folder and start the API using Docker:

```
cd fast_api
docker-compose up --build
docker-compose up
```
Open the web interface in your browser at http://localhost:8000/.

### Interface Features
<p align="center"> <img src="imgs/fast_api_interface.png" alt="t-SNE Visualization of results of autoencoder on Health_and_Personal_Care subdataset"  alt="Fastapi interface screen" width="400">  </p>

- **Product Title:** Enter the title of the product being reviewed.
- **Review Title:** Provide the title of the review.
- **Review Text:** Input the full text of the review for analysis.
- **Model Selection:** Choose an outlier detection model (Isolation Forest, AutoEncoder, GAN (VAE)) from the dropdown.
- **Include Helpful Vote and Rating:** Include helpful votes and rating information in the analysis when you choose a model trained on this additional features.

### Usage Instructions
1. Run the FastAPI server using Docker compose.
2. Open the web interface in your browser.
3. Fill in the form fields with the relevant review details.
3. Select the model you wish to use for outlier detection.
4. Click Submit to get the outlier detection result.
This interface makes it easy to test the API without needing to use curl commands or other API clients.

## Project Notes
This branch is dedicated to model training and API deployment. For a more comprehensive and scalable architecture with model tracking and cloud storage simulation, please refer to the prod_branch.
The code in this branch provides a straightforward, deployable solution for testing core functionalities of outlier detection and distribution shift scoring.