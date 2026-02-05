# Telco Customer Churn Prediction

This project focuses on analyzing customer behavior and predicting churn for a telecommunications company. It includes a complete pipeline from data analysis to a deployed REST API.

## Project Structure
* ml/ — Core Machine Learning service:
    * main.py: FastAPI server for real-time predictions.
    * mlflow.db: Local MLflow Model Registry database.
    * README.md: Specific documentation for the ML module.
* notebooks/ — Jupyter notebooks with EDA and model training experiments.
* myenv/ — Python virtual environment (local only).
* requirements.txt — List of all project dependencies.

## Tech Stack
* Python: Primary programming language.
* FastAPI: High-performance web framework for the API.
* MLflow: Experiment tracking and model management.
* Scikit-learn & XGBoost: Machine learning frameworks used for modeling.
* Uvicorn: ASGI server to run the FastAPI application.

## Quick Start

### 1. Setup Environment
Ensure you have your virtual environment activated:
Command for Windows: .\myenv\Scripts\activate

### 2. Install Dependencies
Command: pip install -r requirements.txt

### 3. Launch the API
Navigate to the ml folder and start the server: 

Command: cd ml

Command: uvicorn main:app --reload

## API Usage
Once the server is running, you can access:
* Interactive UI: http://127.0.0.1:8000/docs
* Prediction Endpoint: POST /predict

## Results
The project compares multiple models (Logistic Regression, Random Forest, XGBoost) using MLflow. The final model is served via the API, providing a churn probability and risk level for each customer.