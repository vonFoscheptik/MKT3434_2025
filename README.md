# Enhanced ML Course GUI
#21067011 Furkan Karstarlı
## Overview
This application is a graphical user interface for training various machine learning models. It supports regression, classification, SVM, and Naive Bayes on multiple datasets with configurable loss functions and hyperparameters.
Python 3.10 is used at that project!

## Features
- **Data Management:** Load built-in datasets (e.g., Iris, Breast Cancer, Boston Housing) or custom CSV files.  
- **Missing Data Handling:** Options include mean, median, most frequent, or constant imputation.  
- **Training Options:** Choose loss functions (MSE, MAE, Huber, Cross-Entropy, Hinge) and set regularization parameters.  
- **Model Tabs:** Separate configuration tabs for Regression, Classification, SVM, and Naive Bayes.  
- **Visualization:** Displays results via scatter plots (regression) or confusion matrices (classification) with performance metrics.

## Requirements
- Python 3.10+
- PyQt6
- scikit-learn
- numpy, pandas
- matplotlib
- TensorFlow

## Installation
1. **Create and activate a virtual environment:**
   ```bash
	python3.10 -m venv myvenv
	```
	```bash
	venv\Scripts\activate
	```
	```bash
	pip install PyQt6 scikit-learn numpy pandas matplotlib tensorflow
	```
	```bash
	py.exe 21067011.py
	```

