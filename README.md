# Airline Sequence Fragility Modeling

This project builds a machine learning pipeline to estimate sequence-level spoilage/disruption risk in airline operations.

## Project Structure

- `src/data`: data loading and preprocessing
- `src/features`: feature engineering
- `src/models`: model training and evaluation
- `src/simulation`: future decision/simulation layer
- `src/utils`: config and helper functions

## Current Stage

- Binary classification: spoiled / partially spoiled vs not spoiled
- Baselines: Logistic Regression, Random Forest
- Time-based train/test split
- Outputs: metrics, feature importance, plots

## Future Direction

- XGBoost
- Calibration
- Hold vs no-hold simulation