# timeseries-anomaly-detection-with-feature-bagging-and-nested-rotations

timeseries-anomaly-detection-with-feature-bagging-and-nested-rotations

This repository contains an anomaly detection pipeline that utilizes various machine learning models, including Autoencoders (AE), Convolutional Autoencoders (Conv-AE), and Long Short-Term Memory (LSTM) models for detecting anomalies in time-series data and enhance the performance by applying Feature Bagging and Nested Rotations.

The project is structured to facilitate efficient data processing, model training, and inference in a scalable and modular way.

Anomaly detection on timeseries using two techniques:

- Feature Bagging
- Nested Rotations

## Feature Bagging

Get a random subset of a set of features

## Nested Rotations

Partition a set of features. Apply PCA on each partition. Apply Rotations to each partition.

## Features

- Multiple Anomaly Detection Models: Supports AE, Conv-AE, LSTM, and LSTM-VAE models.
- Anomaly detection on timeseries using two techniques: Feature Bagging and Nested Rotations
- Pipeline-Based Execution: Modular pipeline for pre-processing, feature bagging, training, and anomaly detection.
- Multiprocessing Support: Enhances performance for large datasets.
- Feature Bagging & Rotation: Increases model robustness by applying transformations.

## References 

- A. Iliopoulos, J. Violos, C. Diou, and I. Varlamis, “Feature bagging with nested rotations (fbnr) for
anomaly detection in multivariate time series,” Future Generation Computer Systems, vol. 163, p. 107 545,
2025, issn: 0167-739X. doi: https://doi.org/10.1016/j.future.2024.107545. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S0167739X24005090.
