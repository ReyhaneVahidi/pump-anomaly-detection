# Pump Anomaly Detection

This project focuses on detecting anomalies in a pump test bench using time-series features extracted from video recordings. The system monitors pump behavior and derives features such as motion intensity, pump cycle timing, and brightness to characterize normal operations.

## Features

Currently extracted features include:

- **Pump timing**: Duration, start and end times of pump cycles
- **Motion-based features**: Mean, RMS, peak count, slopes, rolling slopes, autocorrelation
- **Frequency-based features**: Dominant frequency, spectral centroid, spectral bandwidth
- **Brightness-based features**: Mean, standard deviation, slope
- **Time metadata**: Year, month, day, hour, minute, weekday


## Goal
Build a feature pipeline and anomaly detection model that can identify abnormal pump behavior from video.

## Tech Stack
- Python
- OpenCV
- NumPy
- SciPy
- Pandas

## Project Status
Work in progress.
