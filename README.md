# Enhanced ML Course GUI  
**Student No:** 21067011  
**Name:** Furkan Karstarlı

## Overview

This is a PyQt6-based graphical interface for training and evaluating machine learning models.  
It provides a flexible environment for experimenting with supervised learning, clustering, dimensionality reduction, and cross-validation.

> 🐍 Python version: **3.10**

---

## 🔍 What’s Inside?
A single GUI that lets you:

* **Load, clean & split data** – built-ins *and* your own CSVs.  
* **Train classic ML models** – regression, classification, SVM, Bayes.  
* **Explore data** – PCA, t-SNE, UMAP, K-Means with elbow plots.  
* **Design & fit neural networks** – drag-and-drop layers, optimisers, LR schedules, live metrics, fine-tune ImageNet back-bones, quick DCGAN demo.  
* **Visualise results** – matplotlib canvas **+** real-time log console.

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

## ✨ New in v6

| Area | Additions |
|------|-----------|
| **Neural Networks (tab #6)** | • Dynamic layer list (Dense, Conv2D, MaxPool, LSTM, GRU, Dropout)<br>• Built-in datasets **MNIST / CIFAR-10 / IMDB** (1-click)<br>• Optimisers **Adam / SGD / RMSprop**<br>• LR schedules **Step** & **Exponential Decay**<br>• Regularisation: **Dropout** & **L2**<br>• Live training curves + QTextBrowser log<br>• **Gradient histogram** after fit |
| **Transfer-Learning** | Load **VGG16 / ResNet50 / MobileNetV2**, auto-resizes greyscale input, frozen backbone + trainable head, optional fine-tune (low LR). |
| **Model I/O** | Save ⇢ `.keras` *(or legacy `.h5`)*, Load ⇠ existing model. |
| **GAN demo** | “DCGAN Demo” button (placeholder trainer – ready to extend). |


---

## 📦 Requirements

| Package | Tested Version |
|---------|---------------|
| Python | 3.10.x |
| PyQt6 | ≥ 6.5 |
| NumPy / Pandas | ≥ 1.26 / 2.2 |
| Matplotlib | ≥ 3.8 |
| scikit-learn | ≥ 1.5 |
| TensorFlow | ≥ 2.16 |
| Plotly *(optional)* | ≥ 5.20 |
| umap-learn | ≥ 0.5 |

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
	pip install pyqt6 numpy pandas matplotlib scikit-learn tensorflow plotly umap-learn
	```
	```bash
	py.exe 21067011.py
	```

