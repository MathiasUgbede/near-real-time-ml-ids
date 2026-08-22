# Near Real-Time Machine Learning-Based Intrusion Detection System

Mathias Amade MSc Cyber Security dissertation project developed at Leeds Beckett University.

## Project Overview

This project implements a machine learning-based intrusion detection system using the UNSW-NB15 dataset.

Three machine learning models were evaluated:

- Random Forest
- Gradient Boosting
- Support Vector Machine

Random Forest was selected for the final two-virtual-machine near real-time experiment.

## System Setup

- VM1: Traffic Generator
- VM2: IDS Detector

VM1 sends network-flow records to VM2 using a TCP connection. VM2 classifies each record using the trained Random Forest model.

## Key Results

- Random Forest Accuracy: 89.75%
- Gradient Boosting Accuracy: 89.86%
- SVM Accuracy: 87.93%
- Streamed Random Forest Accuracy: 88.40%
- Median Detector-Side Latency: 26.36 ms
- Average Detector-Side Latency: 28.83 ms
- Throughput: 34.6 flows/second

# Experimental Results

This folder contains selected outputs from the machine learning intrusion detection experiments.

## Files

- `feature_importance_chart.png` – Top selected features from Random Forest feature importance.
- `rf_confusion_matrix.png` – Random Forest confusion matrix.
- `gb_confusion_matrix.png` – Gradient Boosting confusion matrix.
- `svm_confusion_matrix.png` – Support Vector Machine confusion matrix.
- `realtime_latency_histogram.png` – Distribution of detector-side processing latency during the two-VM experiment.
- `realtime_detection_results.txt` – Summary of the streamed Random Forest detection experiment.

## Dataset

The project uses the UNSW-NB15 dataset. The dataset files are not included in this repository.

## Author

Mathias Amade  
MSc Cyber Security  
Leeds Beckett University
