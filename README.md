# EKF-Based GNSS Spoofing Detection

## Overview
This project detects GNSS spoofing using an Extended Kalman Filter (EKF).

## Key Idea
Residuals from EKF are used to detect anomalies caused by spoofing.

## Method
- Simulated trajectory
- Spoof injection
- EKF estimation
- Detection methods:
  - Fixed threshold
  - Adaptive threshold
  - Jump detection

## Results
- Fixed threshold performs best overall
- EKF masking effect reduces detection performance
- Adaptive and jump methods have high precision but low recall

## Failed vs Improved
- Basic EKF smooths errors → hides spoofing
- Improved EKF (velocity model) improves detection

## Files
- trajectory.png → system behavior
- residual.png → anomaly spike
- f1_chart.png → performance comparison
- metrics.csv → numerical results
