
# Benchmarking SGD Variants for Logistic Regression

## Overview
This project implements Vanilla SGD, Momentum SGD, and Adam from scratch to train a logistic regression model on a large-scale sparse dataset.

## How to Run
```bash
pip install numpy
python train.py
```

## Files
- data_generation.py: Synthetic sparse dataset
- model.py: Logistic regression model
- optimizers.py: Optimizers from scratch
- train.py: Training & benchmarking
- report.txt: Empirical analysis

## Key Improvements
- 200 epochs for meaningful convergence analysis
- Standardized learning rate across optimizers
- Quantitative loss, accuracy, and runtime comparison
