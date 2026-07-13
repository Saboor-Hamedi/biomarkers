# Biomarker Analysis Platform

A desktop application for prostate cancer risk classification using differential pulse voltammetry data. The platform trains an ensemble of machine learning models on two hundred electrochemical current measurements per patient and provides interactive risk analysis through a modern desktop interface.

## Overview

The platform processes DPV voltammetry data from Excel files, extracting two hundred current measurements per patient as features. It trains five machine learning models including Logistic Regression, Random Forest, Support Vector Machine, XGBoost, and a Graph Neural Network to predict prostate cancer risk based on a clinically established PSA threshold of four nanograms per milliliter. The trained models are served through a FastAPI backend and visualized in an Electron application built with React.

The training pipeline achieves strong results, with all models reaching ROC-AUC scores above ninety-six percent and validation accuracies above ninety-three percent on a dataset of one thousand patients.

## Project Structure

The project is organized into three main components:

- server/ contains the Python backend, including the FastAPI server (main.py), the model training pipeline (analyse_data.py), the GNN model definition (gnn_model.py), and the trained model artifacts
- src/main/ contains the Electron main process that spawns the Python server and handles inter-process communication
- src/renderer/ contains the React frontend that displays prediction results, model metrics, and visual analytics

## Data

The training data is stored in an Excel file with one thousand patient sheets, each containing two hundred DPV current measurements. A metadata sheet contains reference PSA, AFP, and CA125 concentrations used to define the risk target.

## Setup

Install the Node.js dependencies with npm install. Install the Python dependencies separately in the server directory. The Python packages required include scikit-learn, xgboost, torch, torch-geometric, pandas, numpy, matplotlib, seaborn, fastapi, uvicorn, and openpyxl.

## Training the Models

From the server/analysis/src directory, run:

python analyse_data.py

This loads the DPV data, trains all five models, saves them as pickle files, and generates diagnostic visualizations.

## Running the Application

Start the FastAPI server from the server directory:

python main.py

Or for development with hot reload:

uvicorn main:app --reload --port 8000

Start the Electron application from the project root:

npm run dev

The Electron app spawns the Python server automatically on port 8001 and provides a graphical interface for making predictions, viewing model performance, and exploring patient data.

## API Endpoints

The FastAPI server exposes endpoints for single-patient prediction, batch analysis, population statistics, model metrics, SHAP feature importance, trajectory simulation, and counterfactual analysis. The full API reference is available in the brain/features/api_endpoints.md documentation.

## Documentation

Detailed documentation is available in the brain/ directory, covering the DPV data pipeline, model ensemble, GNN architecture, API endpoints, Electron app architecture, frontend components, and risk analytics.
