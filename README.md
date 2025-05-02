# Enhanced ML Course GUI  
**Student No:** 21067011  
**Name:** Furkan Karstarlı

## Overview

This is a PyQt6-based graphical interface for training and evaluating machine learning models.  
It provides a flexible environment for experimenting with supervised learning, clustering, dimensionality reduction, and cross-validation.

> 🐍 Python version: **3.10**

---

## Features

- ### 📊 Data Management
  - Load built-in datasets: *Iris*, *Breast Cancer*, *Boston Housing*.
  - Or upload custom CSV files.

- ### 🛠️ Missing Data Handling
  - Impute missing values using:
    - Mean
    - Median
    - Most Frequent
    - Constant (user-defined value)

- ### ⚙️ Training Options
  - Choose loss functions: `MSE`, `MAE`, `Huber`, `Cross-Entropy`, `Hinge`
  - Set regularization strength

- ### 🤖 Model Tabs
  - **Regression:** Linear, Decision Tree, SVR
  - **Classification:** Logistic, Decision Tree, SVM, Naive Bayes
  - **SVM:** Kernel, C, epsilon tuning
  - **Naive Bayes:** Prior input, variance smoothing

- ### 📈 Dimensionality Reduction & Clustering
  - **PCA** with explained variance
  - **LDA** with silhouette score
  - **KMeans** with Elbow Method & clustering score
  - **t-SNE** and **UMAP** 2D/3D projections with perplexity slider

- ### 📋 Evaluation
  - K-Fold Cross-Validation (configurable `k`)
  - Displays metrics:
    - Accuracy
    - MSE / RMSE
  - Visual results:
    - Confusion Matrix (classification)
    - Scatter Plot (regression)

- ### 🧠 Eigen Decomposition
  - Computes eigenvectors from a predefined covariance matrix

---

## Requirements
- Python 3.10+
- PyQt6
- scikit-learn
- numpy, pandas
- matplotlib
- TensorFlow
- Pandas
- Plotly
- Umap-learn

## Installation
1. **Create and activate a virtual environment:**
   First of all you should be in your project directory at terminal.
   Or simply you can open terminal at your Visual Code app after opening the project folder.
   Then type these commands respectively... 
   ```bash
	python3.10 -m venv myvenv
	```
	```bash
	.\venv\Scripts\activate  # Windows
	```
	```bash
	pip install pyqt6 numpy pandas matplotlib scikit-learn plotly umap-learn tensorflow
	```
	```bash
	py.exe 21067011.py
	```

