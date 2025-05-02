# =============================================================================
# Enhanced ML Course GUI Application
# Student Name: <Furkan Karstarlı>
# Student ID: <21067011>
# Description: Educational tool for demonstrating regression, classification,
# clustering, dimensionality reduction, and evaluation techniques via PyQt6.
# =============================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
    QGroupBox, QScrollArea, QTextEdit, QStatusBar, QProgressBar,
    QMessageBox, QDialog, QLineEdit, QTextBrowser
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn import datasets, preprocessing, model_selection, metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap.umap_ as umap  # Correct import for UMAP

from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, mean_squared_error, accuracy_score
import plotly.express as px
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# =============================================================================
# Main GUI Class Definition
# =============================================================================
class EnhancedMLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Basic Window Setup
        self.setWindowTitle("Enhanced ML Course GUI")
        self.setGeometry(100, 100, 1600, 900)

        # Central widget setup
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize training data attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_model_type = None

        # Build the GUI sections
        self.create_data_management_section()
        self.create_missing_data_section()
        self.create_training_options_section()
        self.create_model_tabs()
        self.create_visualization_section()
        self.create_status_bar()

    # =========================================================================
    # Section: Model Tabs Setup
    # =========================================================================
    def create_model_tabs(self):
        """Creates the tab layout for different ML models and features."""
        self.tab_widget = QTabWidget()

        # List of all model and analysis tabs
        tabs = [
            ("Regression", self.create_regression_tab),
            ("Classification", self.create_classification_tab),
            ("SVM", self.create_svm_tab),
            ("Naive Bayes", self.create_naive_bayes_tab),
            ("Dim. Reduction & Clustering", self.create_dimensionality_tab)  # Analysis tab
        ]

        # Dynamically create each tab
        for name, constructor in tabs:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            group = constructor()
            tab_layout.addWidget(group)

            if name not in ["Dim. Reduction & Clustering"]:
                train_btn = QPushButton(f"Train {name} Model")
                train_btn.clicked.connect(lambda checked, f=constructor: self.train_model(f))
                tab_layout.addWidget(train_btn)

            # Make tabs scrollable
            scroll = QScrollArea()
            scroll.setWidget(tab)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, name)

        self.layout.addWidget(self.tab_widget)

    # =========================================================================
    # Section: Status Bar
    # =========================================================================
    def create_status_bar(self):
        """Initializes a status bar with a progress bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    # =========================================================================
    # Section: Visualization Area
    # =========================================================================
    def create_visualization_section(self):
        """Creates the lower part of the GUI for plotting and metric display."""
        group = QGroupBox("Results Visualization")
        layout = QHBoxLayout()

        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.metrics_display = QTextBrowser()

        layout.addWidget(self.canvas)
        layout.addWidget(self.metrics_display)
        group.setLayout(layout)
        self.layout.addWidget(group)

    # =========================================================================
    # Section: Training Configuration Options
    # =========================================================================
    def create_training_options_section(self):
        """Configures training loss function and regularization options."""
        group = QGroupBox("Training Options")
        layout = QHBoxLayout()

        self.loss_combo = QComboBox()
        self.loss_combo.addItems([
            "Default Loss",
            "MSE",
            "MAE",
            "Huber Loss",
            "Cross-Entropy",
            "Hinge Loss"
        ])

        self.regularization_spin = QDoubleSpinBox()
        self.regularization_spin.setRange(0.0001, 10)
        self.regularization_spin.setValue(1.0)

        layout.addWidget(QLabel("Loss Function:"))
        layout.addWidget(self.loss_combo)
        layout.addWidget(QLabel("Regularization:"))
        layout.addWidget(self.regularization_spin)

        group.setLayout(layout)
        self.layout.addWidget(group)

    # =========================================================================
    # Section: Missing Data Handling
    # =========================================================================
    def create_missing_data_section(self):
        """Creates the section for handling missing values."""
        group = QGroupBox("Missing Data Handling")
        layout = QHBoxLayout()

        self.imputation_combo = QComboBox()
        self.imputation_combo.addItems([
            "No Imputation",
            "Mean Imputation",
            "Median Imputation",
            "Most Frequent Imputation",
            "Constant Imputation"
        ])

        self.constant_input = QLineEdit("0")
        self.constant_input.setPlaceholderText("Constant Value")

        layout.addWidget(QLabel("Imputation Method:"))
        layout.addWidget(self.imputation_combo)
        layout.addWidget(QLabel("Constant Value:"))
        layout.addWidget(self.constant_input)

        group.setLayout(layout)
        self.layout.addWidget(group)

    # =========================================================================
    # Section: Dataset Loader
    # =========================================================================
    def create_data_management_section(self):
        """Creates the dataset selection and preprocessing interface."""
        group = QGroupBox("Data Management")
        layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset - Regression",
            "Iris Dataset - Classification",
            "Iris Dataset - SVM",
            "Iris Dataset - Naive Bayes",
            "Breast Cancer Dataset - Regression",
            "Breast Cancer Dataset - Classification",
            "Breast Cancer Dataset - SVM",
            "Breast Cancer Dataset - Naive Bayes",
            "Boston Housing Dataset - Regression",
            "Boston Housing Dataset - Classification",
            "Boston Housing Dataset - SVM",
            "Boston Housing Dataset - Naive Bayes"
        ])

        load_btn = QPushButton("Load Dataset")
        load_btn.clicked.connect(self.load_dataset)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"])

        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)

        layout.addWidget(QLabel("Dataset:"))
        layout.addWidget(self.dataset_combo)
        layout.addWidget(load_btn)
        layout.addWidget(QLabel("Scaling:"))
        layout.addWidget(self.scaling_combo)
        layout.addWidget(QLabel("Test Split:"))
        layout.addWidget(self.split_spin)

        group.setLayout(layout)
        self.layout.addWidget(group)
    # =========================================================================
    # Section: Load Dataset Logic
    # =========================================================================
    def load_dataset(self):
        """Loads and prepares dataset based on user selection."""
        try:
            ds_full = self.dataset_combo.currentText()

            # Load custom dataset from CSV
            if ds_full.startswith("Load Custom Dataset"):
                path, _ = QFileDialog.getOpenFileName(
                    self, "Select Dataset", "", "CSV Files (*.csv);;All Files (*)"
                )
                if not path:
                    return
                data = pd.read_csv(path)
                target = self.select_target_column(data.columns)
                if target is None:
                    return
                X = data.drop(target, axis=1)
                y = data[target]

            # Load built-in datasets
            else:
                parts = ds_full.split(" - ")
                base_name = parts[0].strip()
                func_type = parts[1].strip() if len(parts) > 1 else ""

                if base_name == "Iris Dataset":
                    iris = datasets.load_iris()
                    X = iris.data[:, 1:] if func_type == "Regression" else iris.data
                    y = iris.data[:, 0] if func_type == "Regression" else iris.target

                elif base_name == "Breast Cancer Dataset":
                    cancer = datasets.load_breast_cancer()
                    X = cancer.data[:, 1:] if func_type == "Regression" else cancer.data
                    y = cancer.data[:, 0] if func_type == "Regression" else cancer.target

                elif base_name == "Boston Housing Dataset":
                    boston = datasets.load_boston()
                    X = boston.data
                    if func_type == "Regression":
                        y = boston.target
                    else:
                        median_val = np.median(boston.target)
                        y = (boston.target > median_val).astype(int)
                else:
                    raise ValueError("Unsupported dataset selection.")

            # Handle missing data
            X = pd.DataFrame(X)
            y = pd.Series(y)
            imp_method = self.imputation_combo.currentText()

            if imp_method != "No Imputation":
                if imp_method == "Constant Imputation":
                    strategy = "constant"
                    const_val = float(self.constant_input.text())
                    imputer = SimpleImputer(strategy=strategy, fill_value=const_val)
                else:
                    strategy_map = {
                        "Mean Imputation": "mean",
                        "Median Imputation": "median",
                        "Most Frequent Imputation": "most_frequent"
                    }
                    strategy = strategy_map.get(imp_method, "mean")
                    imputer = SimpleImputer(strategy=strategy)

                X = imputer.fit_transform(X)
                X = pd.DataFrame(X)

            # Train-test split
            test_size = self.split_spin.value()
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Apply feature scaling
            scale = self.scaling_combo.currentText()
            if scale != "No Scaling":
                scaler = {
                    "Standard Scaling": preprocessing.StandardScaler(),
                    "Min-Max Scaling": preprocessing.MinMaxScaler(),
                    "Robust Scaling": preprocessing.RobustScaler()
                }.get(scale)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Save to instance variables
            self.X_train = pd.DataFrame(X_train)
            self.X_test = pd.DataFrame(X_test)
            self.y_train = pd.Series(y_train)
            self.y_test = pd.Series(y_test)
            self.status_bar.showMessage(f"Dataset loaded successfully: {ds_full}")

        except Exception as e:
            self.show_error(f"Dataset loading error:\n{str(e)}")

    def select_target_column(self, columns):
        """Dialog for selecting target column from a custom dataset."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dlg)

        combo = QComboBox()
        combo.addItems(list(columns))
        layout.addWidget(combo)

        confirm_btn = QPushButton("Select")
        confirm_btn.clicked.connect(dlg.accept)
        layout.addWidget(confirm_btn)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    # =========================================================================
    # Section: Regression Tab
    # =========================================================================
    def create_regression_tab(self):
        group = QGroupBox("Regression Models")
        layout = QVBoxLayout()

        self.regression_combo = QComboBox()
        self.regression_combo.addItems([
            "Linear Regression",
            "Decision Tree Regression",
            "Support Vector Regression"
        ])
        layout.addWidget(self.regression_combo)

        self.regression_params = QTextEdit()
        self.regression_params.setPlaceholderText("Regression model parameters (optional)")
        layout.addWidget(self.regression_params)

        group.setLayout(layout)
        return group

    # =========================================================================
    # Section: Classification Tab
    # =========================================================================
    def create_classification_tab(self):
        group = QGroupBox("Classification Models")
        layout = QVBoxLayout()

        self.classification_combo = QComboBox()
        self.classification_combo.addItems([
            "Logistic Regression",
            "Decision Tree Classifier",
            "SVM Classifier"
        ])
        layout.addWidget(self.classification_combo)

        self.classification_params = QTextEdit()
        self.classification_params.setPlaceholderText("Classification model parameters (optional)")
        layout.addWidget(self.classification_params)

        group.setLayout(layout)
        return group

    # =========================================================================
    # Section: SVM Tab
    # =========================================================================
    def create_svm_tab(self):
        group = QGroupBox("SVM Configuration")
        layout = QVBoxLayout()

        self.svm_kernel_combo = QComboBox()
        self.svm_kernel_combo.addItems(["linear", "rbf", "poly"])
        layout.addWidget(QLabel("Kernel:"))
        layout.addWidget(self.svm_kernel_combo)

        self.svm_c_spin = QDoubleSpinBox()
        self.svm_c_spin.setRange(0.0001, 10)
        self.svm_c_spin.setValue(1.0)
        layout.addWidget(QLabel("C value:"))
        layout.addWidget(self.svm_c_spin)

        self.svm_epsilon_spin = QDoubleSpinBox()
        self.svm_epsilon_spin.setRange(0.0, 1.0)
        self.svm_epsilon_spin.setValue(0.1)
        layout.addWidget(QLabel("Epsilon (for SVR):"))
        layout.addWidget(self.svm_epsilon_spin)

        group.setLayout(layout)
        return group

    # =========================================================================
    # Section: Naive Bayes Tab
    # =========================================================================
    def create_naive_bayes_tab(self):
        group = QGroupBox("Naive Bayes Configuration")
        layout = QVBoxLayout()

        self.var_smoothing_spin = QDoubleSpinBox()
        self.var_smoothing_spin.setRange(1e-10, 1.0)
        self.var_smoothing_spin.setValue(1e-9)
        layout.addWidget(QLabel("Variance Smoothing:"))
        layout.addWidget(self.var_smoothing_spin)

        self.prior_input = QLineEdit()
        self.prior_input.setPlaceholderText("Enter priors, e.g., 0.3,0.7")
        layout.addWidget(QLabel("Prior Probabilities:"))
        layout.addWidget(self.prior_input)

        group.setLayout(layout)
        return group
    # =========================================================================
    # Section: Dimensionality Reduction & Clustering Tab
    # =========================================================================
    def create_dimensionality_tab(self):
        group = QGroupBox("Dimensionality Reduction & Clustering")
        layout = QVBoxLayout()

        # PCA controls
        pca_layout = QHBoxLayout()
        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(1, 10)
        self.pca_components_spin.setValue(2)
        pca_btn = QPushButton("Run PCA")
        pca_btn.clicked.connect(self.run_pca)
        pca_layout.addWidget(QLabel("PCA Components:"))
        pca_layout.addWidget(self.pca_components_spin)
        pca_layout.addWidget(pca_btn)

        # LDA
        lda_btn = QPushButton("Run LDA (Supervised)")
        lda_btn.clicked.connect(self.run_lda)

        # Clustering
        cluster_layout = QHBoxLayout()
        self.kmeans_k_spin = QSpinBox()
        self.kmeans_k_spin.setRange(2, 10)
        self.kmeans_k_spin.setValue(3)
        elbow_btn = QPushButton("Elbow Method")
        elbow_btn.clicked.connect(self.run_elbow_method)
        kmeans_btn = QPushButton("Run KMeans")
        kmeans_btn.clicked.connect(self.run_kmeans)
        cluster_layout.addWidget(QLabel("k Clusters:"))
        cluster_layout.addWidget(self.kmeans_k_spin)
        cluster_layout.addWidget(elbow_btn)
        cluster_layout.addWidget(kmeans_btn)

        # t-SNE & UMAP
        proj_layout = QHBoxLayout()
        self.perplexity_spin = QDoubleSpinBox()
        self.perplexity_spin.setRange(5.0, 50.0)
        self.perplexity_spin.setValue(30.0)
        tsne_btn = QPushButton("Run t-SNE")
        tsne_btn.clicked.connect(self.run_tsne)
        umap_btn = QPushButton("Run UMAP")
        umap_btn.clicked.connect(self.run_umap)
        proj_layout.addWidget(QLabel("Perplexity:"))
        proj_layout.addWidget(self.perplexity_spin)
        proj_layout.addWidget(tsne_btn)
        proj_layout.addWidget(umap_btn)

        # K-Fold CV
        cv_layout = QHBoxLayout()
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 10)
        self.kfold_spin.setValue(5)
        cv_btn = QPushButton("Run K-Fold CV")
        cv_btn.clicked.connect(self.run_cross_validation)
        cv_layout.addWidget(QLabel("Folds:"))
        cv_layout.addWidget(self.kfold_spin)
        cv_layout.addWidget(cv_btn)

        # Eigen computation
        eig_btn = QPushButton("Compute Σ Eigenvectors")
        eig_btn.clicked.connect(self.compute_cov_eig)

        layout.addLayout(pca_layout)
        layout.addWidget(lda_btn)
        layout.addLayout(cluster_layout)
        layout.addLayout(proj_layout)
        layout.addLayout(cv_layout)
        layout.addWidget(eig_btn)
        group.setLayout(layout)
        return group

    # =========================================================================
    # Section: Dimensionality & Clustering Methods
    # =========================================================================
    def run_pca(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return
        n = self.pca_components_spin.value()
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(self.X_train)
        fig = px.scatter_matrix(pd.DataFrame(X_pca), dimensions=range(n),
                                title=f"PCA (Variance: {np.sum(pca.explained_variance_ratio_):.2f})")
        fig.show()

    def run_lda(self):
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load a dataset first.")
            return
        try:
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(self.X_train, self.y_train)
            score = silhouette_score(X_lda, self.y_train)
            fig = px.scatter(x=X_lda[:, 0], y=X_lda[:, 1],
                             color=self.y_train.astype(str),
                             title=f"LDA Projection (Silhouette: {score:.2f})")
            fig.show()
        except Exception as e:
            self.show_error(f"LDA failed: {str(e)}")

    def run_elbow_method(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return
        distortions = []
        for k in range(1, 10):
            km = KMeans(n_clusters=k, n_init='auto')
            km.fit(self.X_train)
            distortions.append(km.inertia_)
        plt.plot(range(1, 10), distortions, 'bx-')
        plt.title("Elbow Method for k")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.show()

    def run_kmeans(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return
        k = self.kmeans_k_spin.value()
        km = KMeans(n_clusters=k, n_init='auto')
        y_km = km.fit_predict(self.X_train)
        score = silhouette_score(self.X_train, y_km)
        fig = px.scatter_matrix(pd.DataFrame(self.X_train), color=y_km.astype(str),
                                title=f"KMeans Clustering (Silhouette: {score:.2f})")
        fig.show()

    def run_tsne(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return
        tsne = TSNE(n_components=2, perplexity=self.perplexity_spin.value(), random_state=42)
        X_tsne = tsne.fit_transform(self.X_train)
        fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], title="t-SNE Projection")
        fig.show()

    def run_umap(self):
        if self.X_train is None:
            self.show_error("Please load a dataset first.")
            return
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        X_umap = reducer.fit_transform(self.X_train)
        fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], title="UMAP Projection")
        fig.show()

    def run_cross_validation(self):
        try:
            if self.X_train is None or self.y_train is None:
                self.show_error("Please load a dataset first.")
                return

            k = self.kfold_spin.value()
            kf = KFold(n_splits=k, shuffle=True, random_state=42)

            is_regression = np.issubdtype(self.y_train.dtype, np.floating)
            model = LinearRegression() if is_regression else LogisticRegression(max_iter=1000)
            accs, mses, rmses = [], [], []

            for train_idx, val_idx in kf.split(self.X_train):
                X_tr = self.X_train.iloc[train_idx] if hasattr(self.X_train, 'iloc') else self.X_train[train_idx]
                X_val = self.X_train.iloc[val_idx] if hasattr(self.X_train, 'iloc') else self.X_train[val_idx]
                y_tr = self.y_train.iloc[train_idx] if hasattr(self.y_train, 'iloc') else self.y_train[train_idx]
                y_val = self.y_train.iloc[val_idx] if hasattr(self.y_train, 'iloc') else self.y_train[val_idx]

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)

                if is_regression:
                    mse = mean_squared_error(y_val, y_pred)
                    rmse = np.sqrt(mse)
                    mses.append(mse)
                    rmses.append(rmse)
                else:
                    acc = accuracy_score(y_val, y_pred)
                    accs.append(acc)

            if is_regression:
                msg = f"Mean MSE: {np.mean(mses):.4f}\nMean RMSE: {np.mean(rmses):.4f}"
            else:
                msg = f"Mean Accuracy: {np.mean(accs):.4f}\nStd Dev: {np.std(accs):.4f}"

            QMessageBox.information(self, "Cross-Validation Results", msg)

        except Exception as e:
            self.show_error(f"K-Fold Error:\n{str(e)}")

    def compute_cov_eig(self):
        """Computes eigenvectors of a fixed covariance matrix Σ."""
        sigma = np.array([[5, 2], [2, 3]])
        eigvals, eigvecs = np.linalg.eig(sigma)
        msg = f"Σ = [[5, 2], [2, 3]]\n\nEigenvalues:\n{eigvals}\n\nEigenvectors:\n{eigvecs}"
        QMessageBox.information(self, "Eigen Decomposition", msg)

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

# ============================================================================
# Main Application Launcher
# ============================================================================
def main():
    app = QApplication(sys.argv)
    window = EnhancedMLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
# =============================================================================
# End of Enhanced ML Course GUI Application
# =============================================================================
