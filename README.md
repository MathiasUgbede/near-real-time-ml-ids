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

## Files

- `MSc_Cybersecurity_Dissertation.pdf` – Final MSc dissertation
- `scripts/` – Python implementation scripts

## Dataset

The project uses the UNSW-NB15 dataset. The dataset files are not included in this repository.

## Author

Mathias Amade  
MSc Cyber Security  
Leeds Beckett University
