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
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class EnhancedMLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced ML Course GUI")
        self.setGeometry(100, 100, 1600, 900)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_model_type = None
        
        self.create_data_management_section()
        self.create_missing_data_section()
        self.create_training_options_section()
        self.create_model_tabs()
        self.create_visualization_section()
        self.create_status_bar()
    
    def create_data_management_section(self):
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
    
    def create_missing_data_section(self):
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
    
    def create_training_options_section(self):
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
    
    def create_model_tabs(self):
        self.tab_widget = QTabWidget()
        tabs = [
            ("Regression", self.create_regression_tab),
            ("Classification", self.create_classification_tab),
            ("SVM", self.create_svm_tab),
            ("Naive Bayes", self.create_naive_bayes_tab)
        ]
        for name, func in tabs:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            group = func()
            tab_layout.addWidget(group)
            train_btn = QPushButton(f"Train {name} Model")
            train_btn.clicked.connect(lambda checked, f=func: self.train_model(f))
            tab_layout.addWidget(train_btn)
            scroll = QScrollArea()
            scroll.setWidget(tab)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, name)
        self.layout.addWidget(self.tab_widget)
    
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
        self.regression_params.setPlaceholderText("Regression model parameters (if any)")
        layout.addWidget(self.regression_params)
        group.setLayout(layout)
        return group
    
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
        self.classification_params.setPlaceholderText("Classification model parameters (if any)")
        layout.addWidget(self.classification_params)
        group.setLayout(layout)
        return group
    
    def create_svm_tab(self):
        group = QGroupBox("SVM Configuration")
        layout = QVBoxLayout()
        self.svm_kernel_combo = QComboBox()
        self.svm_kernel_combo.addItems(["linear", "rbf", "poly"])
        layout.addWidget(QLabel("Kernel:"))
        layout.addWidget(self.svm_kernel_combo)
        c_layout = QHBoxLayout()
        c_layout.addWidget(QLabel("C:"))
        self.svm_c_spin = QDoubleSpinBox()
        self.svm_c_spin.setRange(0.0001, 10)
        self.svm_c_spin.setValue(1.0)
        c_layout.addWidget(self.svm_c_spin)
        layout.addLayout(c_layout)
        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("Epsilon (for SVR):"))
        self.svm_epsilon_spin = QDoubleSpinBox()
        self.svm_epsilon_spin.setRange(0.0, 1.0)
        self.svm_epsilon_spin.setValue(0.1)
        eps_layout.addWidget(self.svm_epsilon_spin)
        layout.addLayout(eps_layout)
        group.setLayout(layout)
        return group
    
    def create_naive_bayes_tab(self):
        group = QGroupBox("Naive Bayes Configuration")
        layout = QVBoxLayout()
        vs_layout = QHBoxLayout()
        vs_layout.addWidget(QLabel("Var Smoothing:"))
        self.var_smoothing_spin = QDoubleSpinBox()
        self.var_smoothing_spin.setRange(1e-10, 1)
        self.var_smoothing_spin.setValue(1e-9)
        vs_layout.addWidget(self.var_smoothing_spin)
        layout.addLayout(vs_layout)
        prior_layout = QHBoxLayout()
        prior_layout.addWidget(QLabel("Prior Probabilities:"))
        self.prior_input = QLineEdit()
        self.prior_input.setPlaceholderText("e.g., 0.3,0.7")
        prior_layout.addWidget(self.prior_input)
        layout.addLayout(prior_layout)
        group.setLayout(layout)
        return group
    
    def create_visualization_section(self):
        group = QGroupBox("Results Visualization")
        layout = QHBoxLayout()
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.metrics_display = QTextBrowser()
        layout.addWidget(self.metrics_display)
        group.setLayout(layout)
        self.layout.addWidget(group)
    
    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def load_dataset(self):
        try:
            ds_full = self.dataset_combo.currentText()
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
            else:
                parts = ds_full.split(" - ")
                base_name = parts[0].strip()
                func_type = parts[1].strip() if len(parts) > 1 else ""
                if base_name == "Iris Dataset":
                    iris = datasets.load_iris()
                    if func_type == "Regression":
                        X = iris.data[:, 1:]
                        y = iris.data[:, 0]
                    else:
                        X = iris.data
                        y = iris.target
                elif base_name == "Breast Cancer Dataset":
                    cancer = datasets.load_breast_cancer()
                    if func_type == "Regression":
                        X = cancer.data[:, 1:]
                        y = cancer.data[:, 0]
                    else:
                        X = cancer.data
                        y = cancer.target
                elif base_name == "Boston Housing Dataset":
                    boston = datasets.load_boston()
                    if func_type == "Regression":
                        X = boston.data
                        y = boston.target
                    else:
                        X = boston.data
                        median_val = np.median(boston.target)
                        y = (boston.target > median_val).astype(int)
                else:
                    raise ValueError("Unsupported dataset")
            X = pd.DataFrame(X)
            y = pd.Series(y)
            imp_method = self.imputation_combo.currentText()
            if imp_method != "No Imputation":
                if imp_method == "Mean Imputation":
                    strategy = "mean"
                elif imp_method == "Median Imputation":
                    strategy = "median"
                elif imp_method == "Most Frequent Imputation":
                    strategy = "most_frequent"
                elif imp_method == "Constant Imputation":
                    strategy = "constant"
                    const_val = float(self.constant_input.text())
                    imp = SimpleImputer(strategy=strategy, fill_value=const_val)
                    X = imp.fit_transform(X)
                    X = pd.DataFrame(X)
                else:
                    strategy = "mean"
                if imp_method != "Constant Imputation":
                    imp = SimpleImputer(strategy=strategy)
                    X = imp.fit_transform(X)
                    X = pd.DataFrame(X)
            test_size = self.split_spin.value()
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            scale = self.scaling_combo.currentText()
            if scale != "No Scaling":
                if scale == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scale == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scale == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.status_bar.showMessage(f"Loaded {ds_full} successfully")
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def select_target_column(self, cols):
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dlg)
        combo = QComboBox()
        combo.addItems(list(cols))
        layout.addWidget(combo)
        btn = QPushButton("Select")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def train_model(self, create_func):
        try:
            if self.X_train is None or self.y_train is None:
                self.show_error("Please load a dataset first")
                return
            loss = self.loss_combo.currentText()
            reg = self.regularization_spin.value()
            if create_func == self.create_regression_tab:
                self.current_model_type = "Regression"
                model_type = self.regression_combo.currentText()
                if model_type == "Linear Regression":
                    model = Pipeline([
                        ('poly', PolynomialFeatures(degree=2)),
                        ('linear', LinearRegression(fit_intercept=True))
                    ])
                elif model_type == "Decision Tree Regression":
                    model = DecisionTreeRegressor(max_depth=5)
                elif model_type == "Support Vector Regression":
                    model = SVR(
                        kernel=self.svm_kernel_combo.currentText(),
                        C=self.svm_c_spin.value(),
                        epsilon=self.svm_epsilon_spin.value()
                    )
            elif create_func == self.create_classification_tab:
                self.current_model_type = "Classification"
                model_type = self.classification_combo.currentText()
                if model_type == "Logistic Regression":
                    model = LogisticRegression(C=reg, multi_class='ovr', max_iter=1000)
                elif model_type == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(max_depth=5)
                elif model_type == "SVM Classifier":
                    model = SVC(
                        kernel=self.svm_kernel_combo.currentText(),
                        C=self.svm_c_spin.value()
                    )
            elif create_func == self.create_svm_tab:
                self.current_model_type = "Classification"
                model = SVC(
                    kernel=self.svm_kernel_combo.currentText(),
                    C=self.svm_c_spin.value()
                )
            elif create_func == self.create_naive_bayes_tab:
                self.current_model_type = "Classification"
                priors_text = self.prior_input.text().strip()
                priors = None
                if priors_text:
                    try:
                        priors = [float(x) for x in priors_text.split(',')]
                    except:
                        priors = None
                model = GaussianNB(var_smoothing=self.var_smoothing_spin.value(), priors=priors)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.current_model = model
            self.visualize_results(y_pred)
            self.status_bar.showMessage(f"Trained model successfully")
        except Exception as e:
            self.show_error(f"Model training error: {str(e)}")
    
    def visualize_results(self, y_pred):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.current_model_type == "Regression":
            ax.scatter(self.y_test, y_pred, color='blue', alpha=0.7)
            ax.plot([min(self.y_test), max(self.y_test)],
                    [min(self.y_test), max(self.y_test)], 'r--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Regression: Actual vs Predicted")
            mse = metrics.mean_squared_error(self.y_test, y_pred)
            mae = metrics.mean_absolute_error(self.y_test, y_pred)
            r2 = metrics.r2_score(self.y_test, y_pred)
            txt = f"Regression Metrics:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"
        else:
            cm = metrics.confusion_matrix(self.y_test, y_pred)
            im = ax.imshow(cm, cmap=plt.cm.Blues)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]),
                            ha='center', va='center', color='black')
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            self.figure.colorbar(im, ax=ax)
            acc = metrics.accuracy_score(self.y_test, y_pred)
            prec = metrics.precision_score(self.y_test, y_pred, average='weighted')
            rec = metrics.recall_score(self.y_test, y_pred, average='weighted')
            f1 = metrics.f1_score(self.y_test, y_pred, average='weighted')
            txt = (
                f"Classification Metrics:\n"
                f"Accuracy: {acc:.4f}\n"
                f"Precision: {prec:.4f}\n"
                f"Recall: {rec:.4f}\n"
                f"F1 Score: {f1:.4f}"
            )
        self.canvas.draw()
        self.metrics_display.setText(txt)
    
    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

def main():
    app = QApplication(sys.argv)
    window = EnhancedMLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
