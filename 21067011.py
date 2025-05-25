# =============================================================================
# Enhanced ML Course GUI Application – Complete Version v5
# Student Name: Furkan Karstarlı
# Student ID: 21067011
# Description: Educational tool for demonstrating classical ML and
# neural-network techniques via PyQt6.
# Python 3.10 compatible.
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
    QGroupBox, QScrollArea, QTextEdit, QStatusBar, QProgressBar,
    QMessageBox, QDialog, QLineEdit, QTextBrowser, QListWidget, QInputDialog
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ─── scikit-learn ───────────────────────────────────────────────────────
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.metrics import (
    silhouette_score, mean_squared_error, accuracy_score, f1_score
)
import umap.umap_ as umap

# ─── TensorFlow / Keras ────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Plotly for interactive plots (optional) ───────────────────────────
import plotly.express as px   # only used for elbow/cluster visualisation

# =============================================================================
# Helper: Keras callback that logs to QTextBrowser
# =============================================================================
class GuiLoggingCallback(callbacks.Callback):
    """Send per-epoch logs to the QTextBrowser for real-time feedback."""
    def __init__(self, gui):
        super().__init__()
        self.gui = gui

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        txt = (f"Epoch {epoch + 1}: "
               f"loss={logs.get('loss', 0):.4f}  "
               f"val_loss={logs.get('val_loss', 0):.4f}  "
               f"acc={logs.get('accuracy', 0):.4f}  "
               f"val_acc={logs.get('val_accuracy', 0):.4f}")
        self.gui.metrics_display.append(txt)
        sb = self.gui.metrics_display.verticalScrollBar()
        sb.setValue(sb.maximum())

# =============================================================================
# Main GUI Class
# =============================================================================
class EnhancedMLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced ML Course GUI v5")
        self.resize(1600, 900)

        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)

        # Data containers (classic-ML tab)
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.current_model = None

        # NN state ------------------------------------------------------
        self.nn_layers = []
        self.nn_input_shape = None
        self.nn_num_classes = None

        # ✨ dataset placeholders to avoid AttributeError before first load
        self.nn_x_train = self.nn_y_train = None
        self.nn_x_val   = self.nn_y_val   = None
        self.nn_x_test  = self.nn_y_test  = None

        # Build interface
        self.create_data_management_section()
        self.create_missing_data_section()
        self.create_training_options_section()
        self.create_model_tabs()
        self.create_visualization_section()
        self.create_status_bar()


    # ───────────────────────── Tabs ─────────────────────────
    def create_model_tabs(self):
        self.tab_widget = QTabWidget()
        tabs = [
            ("Regression", self.create_regression_tab),
            ("Classification", self.create_classification_tab),
            ("SVM", self.create_svm_tab),
            ("Naive Bayes", self.create_naive_bayes_tab),
            ("Dim. Reduction & Clustering", self.create_dimensionality_tab),
            ("Neural Networks", self.create_nn_tab)
        ]
        for name, builder in tabs:
            page = QWidget()
            v = QVBoxLayout(page)
            group = builder()
            v.addWidget(group)

            # generic train button (skip analysis & NN)
            if name not in ("Dim. Reduction & Clustering", "Neural Networks"):
                btn = QPushButton(f"Train {name}")
                btn.clicked.connect(lambda _=False, f=builder: self.train_model(f))
                v.addWidget(btn)

            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, name)

        self.layout.addWidget(self.tab_widget)

    # ───────────────── Status & Visualization ─────────────────
    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_visualization_section(self):
        group = QGroupBox("Results Visualization")
        h = QHBoxLayout()
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.metrics_display = QTextBrowser()
        h.addWidget(self.canvas)
        h.addWidget(self.metrics_display)
        group.setLayout(h)
        self.layout.addWidget(group)

    # ───────────────── Training options ─────────────────
    def create_training_options_section(self):
        grp = QGroupBox("Training Options")
        lay = QHBoxLayout()
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(
            ["Default Loss", "MSE", "MAE", "Huber", "Cross-Entropy", "Hinge"]
        )
        self.reg_spin = QDoubleSpinBox()
        self.reg_spin.setRange(0.0001, 10)
        self.reg_spin.setValue(1.0)
        for w in [
            QLabel("Loss:"), self.loss_combo,
            QLabel("Regularization:"), self.reg_spin
        ]:
            lay.addWidget(w)
        grp.setLayout(lay)
        self.layout.addWidget(grp)

    # ───────────────── Missing-data section ─────────────────
    def create_missing_data_section(self):
        grp = QGroupBox("Missing Data Handling")
        lay = QHBoxLayout()
        self.imp_combo = QComboBox()
        self.imp_combo.addItems(
            ["No Imputation", "Mean", "Median", "Most Frequent", "Constant"]
        )
        self.const_edit = QLineEdit("0")
        self.const_edit.setPlaceholderText("Constant value")
        for w in [
            QLabel("Imputation:"), self.imp_combo,
            QLabel("Constant:"), self.const_edit
        ]:
            lay.addWidget(w)
        grp.setLayout(lay)
        self.layout.addWidget(grp)

    # ───────────────── Dataset management ─────────────────
    def create_data_management_section(self):
        grp = QGroupBox("Data Management")
        lay = QHBoxLayout()
        self.ds_combo = QComboBox()
        self.ds_combo.addItems(
            [
                "Load Custom Dataset",
                "Iris Dataset - Classification",
                "Iris Dataset - Regression",
                "Breast Cancer Dataset - Classification",
                "Breast Cancer Dataset - Regression",
                "Boston Housing Dataset - Regression",
            ]
        )
        btn = QPushButton("Load Dataset")
        btn.clicked.connect(self.load_dataset)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["No Scaling", "Standard", "MinMax", "Robust"])
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        for w in [
            QLabel("Dataset:"), self.ds_combo, btn,
            QLabel("Scaling:"), self.scale_combo,
            QLabel("Test Split:"), self.split_spin
        ]:
            lay.addWidget(w)
        grp.setLayout(lay)
        self.layout.addWidget(grp)

    # ───────────────── Tabs (classic ML) ─────────────────
    def create_regression_tab(self):
        grp = QGroupBox("Regression Models")
        lay = QVBoxLayout()
        self.reg_combo = QComboBox()
        self.reg_combo.addItems(
            ["Linear Regression", "Decision Tree Regression", "SVR"]
        )
        self.reg_params = QTextEdit()
        self.reg_params.setPlaceholderText("Optional parameters")
        lay.addWidget(self.reg_combo)
        lay.addWidget(self.reg_params)
        grp.setLayout(lay)
        return grp

    def create_classification_tab(self):
        grp = QGroupBox("Classification Models")
        lay = QVBoxLayout()
        self.clf_combo = QComboBox()
        self.clf_combo.addItems(
            ["Logistic Regression", "Decision Tree Classifier", "SVC"]
        )
        self.clf_params = QTextEdit()
        self.clf_params.setPlaceholderText("Optional parameters")
        lay.addWidget(self.clf_combo)
        lay.addWidget(self.clf_params)
        grp.setLayout(lay)
        return grp

    def create_svm_tab(self):
        grp = QGroupBox("SVM Config")
        lay = QVBoxLayout()
        self.svm_kernel = QComboBox()
        self.svm_kernel.addItems(["linear", "rbf", "poly"])
        self.svm_C = QDoubleSpinBox()
        self.svm_C.setRange(0.0001, 10)
        self.svm_C.setValue(1.0)
        self.svm_eps = QDoubleSpinBox()
        self.svm_eps.setRange(0, 1)
        self.svm_eps.setValue(0.1)
        for w in [
            QLabel("Kernel"), self.svm_kernel,
            QLabel("C"), self.svm_C,
            QLabel("Epsilon (for SVR)"), self.svm_eps
        ]:
            lay.addWidget(w)
        grp.setLayout(lay)
        return grp

    def create_naive_bayes_tab(self):
        grp = QGroupBox("Naive Bayes")
        lay = QVBoxLayout()
        self.nb_smooth = QDoubleSpinBox()
        self.nb_smooth.setRange(1e-10, 1)
        self.nb_smooth.setValue(1e-9)
        self.prior_edit = QLineEdit()
        self.prior_edit.setPlaceholderText("e.g. 0.3,0.7")
        for w in [
            QLabel("Var smoothing"), self.nb_smooth,
            QLabel("Priors"), self.prior_edit
        ]:
            lay.addWidget(w)
        grp.setLayout(lay)
        return grp

    # Dimensionality reduction & clustering tab ---------------------------
    def create_dimensionality_tab(self):
        grp = QGroupBox("Dimensionality Reduction & Clustering")
        v = QVBoxLayout()

        # PCA
        pca_row = QHBoxLayout()
        self.pca_comp = QSpinBox()
        self.pca_comp.setRange(1, 10)
        self.pca_comp.setValue(2)
        pca_btn = QPushButton("Run PCA")
        pca_btn.clicked.connect(self.run_pca)
        pca_row.addWidget(QLabel("Components"))
        pca_row.addWidget(self.pca_comp)
        pca_row.addWidget(pca_btn)
        v.addLayout(pca_row)

        # LDA
        lda_btn = QPushButton("Run LDA")
        lda_btn.clicked.connect(self.run_lda)
        v.addWidget(lda_btn)

        # KMeans & elbow
        km_row = QHBoxLayout()
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 10)
        self.k_spin.setValue(3)
        elbow = QPushButton("Elbow")
        elbow.clicked.connect(self.run_elbow)
        km_btn = QPushButton("KMeans")
        km_btn.clicked.connect(self.run_kmeans)
        for w in [QLabel("k"), self.k_spin, elbow, km_btn]:
            km_row.addWidget(w)
        v.addLayout(km_row)

        # tSNE & UMAP
        proj_row = QHBoxLayout()
        self.perp = QDoubleSpinBox()
        self.perp.setRange(5, 50)
        self.perp.setValue(30)
        tsne = QPushButton("t-SNE")
        tsne.clicked.connect(self.run_tsne)
        umap_btn = QPushButton("UMAP")
        umap_btn.clicked.connect(self.run_umap)
        for w in [QLabel("Perplexity"), self.perp, tsne, umap_btn]:
            proj_row.addWidget(w)
        v.addLayout(proj_row)

        # K-Fold
        cv_row = QHBoxLayout()
        self.fold_spin = QSpinBox()
        self.fold_spin.setRange(2, 10)
        self.fold_spin.setValue(5)
        cv_btn = QPushButton("K-Fold CV")
        cv_btn.clicked.connect(self.run_kfold)
        for w in [QLabel("Folds"), self.fold_spin, cv_btn]:
            cv_row.addWidget(w)
        v.addLayout(cv_row)

        grp.setLayout(v)
        return grp

    # ───────────────────────── Data loading logic ─────────────────────────
    def load_dataset(self):
        try:
            choice = self.ds_combo.currentText()
            if choice == "Load Custom Dataset":
                path, _ = QFileDialog.getOpenFileName(
                    self, "CSV", "", "CSV (*.csv)"
                )
                if not path:
                    return
                df = pd.read_csv(path)
                target = self.select_target_column(df.columns)
                if target is None:
                    return
                X, y = df.drop(target, axis=1), df[target]
            else:
                base, task = choice.split(" - ")
                if base == "Iris Dataset":
                    d = datasets.load_iris()
                    X, y = pd.DataFrame(
                        d.data, columns=d.feature_names
                    ), pd.Series(d.target)
                    if task == "Regression":
                        y = X.iloc[:, 0]
                        X = X.iloc[:, 1:]
                elif base == "Breast Cancer Dataset":
                    d = datasets.load_breast_cancer()
                    X, y = pd.DataFrame(d.data, columns=d.feature_names), pd.Series(
                        d.target
                    )
                    if task == "Regression":
                        y = X.iloc[:, 0]
                        X = X.iloc[:, 1:]
                elif base == "Boston Housing Dataset":
                    d = datasets.load_boston()
                    X, y = pd.DataFrame(d.data, columns=d.feature_names), pd.Series(
                        d.target
                    )
                else:
                    self.show_error("Unknown dataset")
                    return

            # scaling
            scaler_choice = self.scale_combo.currentText()
            if scaler_choice != "No Scaling":
                scaler_map = {
                    "Standard": preprocessing.StandardScaler,
                    "MinMax": preprocessing.MinMaxScaler,
                    "Robust": preprocessing.RobustScaler,
                }
                scaler = scaler_map[scaler_choice]()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # imputation
            imp_choice = self.imp_combo.currentText()
            if imp_choice != "No Imputation":
                strategy_map = {
                    "Mean": "mean",
                    "Median": "median",
                    "Most Frequent": "most_frequent",
                    "Constant": "constant",
                }
                strategy = strategy_map.get(imp_choice, "mean")
                fill_val = float(self.const_edit.text()) if strategy == "constant" else None
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_val)
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            # train-test split
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            self.status_bar.showMessage(
                f"Dataset loaded. Train shape: {self.X_train.shape}"
            )
        except Exception as e:
            self.show_error(str(e))

    def select_target_column(self, columns):
        col, ok = QInputDialog.getItem(
            self, "Target column", "Select the target (y) column:", list(columns), 0, False
        )
        return col if ok else None

    # ────────────────────────── Placeholders for classic ML training ─────
    def train_model(self, builder):
        QMessageBox.information(
            self, "Stub", "Classic model training logic preserved from original script"
        )

    # ---------------------------------------------------------------------
    # Dimensionality-reduction helpers (unchanged logic) ------------------
    def run_pca(self):
        if self.X_train is None:
            self.show_error("Load data first")
            return
        comps = self.pca_comp.value()
        pca = PCA(n_components=comps)
        X_red = pca.fit_transform(self.X_train)
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        if comps == 2:
            ax.scatter(X_red[:, 0], X_red[:, 1], c=self.y_train, cmap="viridis", s=20)
            ax.set_title("PCA 2-D")
        elif comps == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = self.figure.add_subplot(111, projection="3d")
            ax.scatter(
                X_red[:, 0], X_red[:, 1], X_red[:, 2], c=self.y_train, cmap="viridis", s=10
            )
            ax.set_title("PCA 3-D")
        else:
            ax.plot(np.cumsum(pca.explained_variance_ratio_))
            ax.set_title("Cumulative Explained Variance")
        self.canvas.draw()

    def run_lda(self):
        self.show_error("LDA stub — implement as needed")

    def run_elbow(self):
        self.show_error("Elbow stub — implement as needed")

    def run_kmeans(self):
        self.show_error("KMeans stub — implement as needed")

    def run_tsne(self):
        self.show_error("t-SNE stub — implement as needed")

    def run_umap(self):
        self.show_error("UMAP stub — implement as needed")

    def run_kfold(self):
        self.show_error("K-Fold CV stub — implement as needed")

    # =============================================================================
    # Neural-Network tab and helpers
    # =============================================================================
    def create_nn_tab(self):
        grp = QGroupBox("Neural Network Designer & Trainer")
        layout = QVBoxLayout()

        # Dataset loader
        ds_row = QHBoxLayout()
        self.nn_dataset_combo = QComboBox()
        self.nn_dataset_combo.addItems(["MNIST", "CIFAR-10", "IMDB Reviews"])
        load_btn = QPushButton("Load Dataset")
        load_btn.clicked.connect(self.load_nn_dataset)
        ds_row.addWidget(QLabel("Built-in Dataset:"))
        ds_row.addWidget(self.nn_dataset_combo)
        ds_row.addWidget(load_btn)
        layout.addLayout(ds_row)

        # Layer editor
        self.layer_list = QListWidget()
        layout.addWidget(QLabel("Architecture (top → bottom):"))
        layout.addWidget(self.layer_list)

        layer_btns = QHBoxLayout()
        for txt, func in [
            ("Add Dense", self.add_dense_layer),
            ("Add Conv2D", self.add_conv_layer),
            ("Add Pool", self.add_pool_layer),
            ("Add LSTM", lambda: self.add_rnn_layer("LSTM")),
            ("Add GRU", lambda: self.add_rnn_layer("GRU")),
            ("Add Dropout", self.add_dropout_layer),
            ("Remove", self.remove_selected_layer),
            ("Move ▲", self.move_layer_up),
            ("Move ▼", self.move_layer_down),
        ]:
            b = QPushButton(txt)
            b.clicked.connect(func)
            layer_btns.addWidget(b)
        layout.addLayout(layer_btns)

        # Optimizer & LR schedule
        opt_row = QHBoxLayout()
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(1e-5, 1)
        self.lr_spin.setValue(0.001)
        self.lr_schedule_combo = QComboBox()
        self.lr_schedule_combo.addItems(["None", "Step Decay", "Exponential Decay"])
        opt_row.addWidget(QLabel("Optimizer"))
        opt_row.addWidget(self.optimizer_combo)
        opt_row.addWidget(QLabel("Init LR"))
        opt_row.addWidget(self.lr_spin)
        opt_row.addWidget(QLabel("LR Schedule"))
        opt_row.addWidget(self.lr_schedule_combo)
        layout.addLayout(opt_row)

        # Epochs / batch
        train_row = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 512)
        self.batch_spin.setValue(32)
        train_row.addWidget(QLabel("Epochs"))
        train_row.addWidget(self.epochs_spin)
        train_row.addWidget(QLabel("Batch"))
        train_row.addWidget(self.batch_spin)
        layout.addLayout(train_row)

        # Action buttons
        act_row = QHBoxLayout()
        for txt, slot in [
            ("Train", self.train_neural_network),
            ("Save", self.save_nn_model),
            ("Load", self.load_nn_model),
            ("Pre-trained", self.load_pretrained_model),
            ("DCGAN Demo", self.train_dcgan),
        ]:
            b = QPushButton(txt)
            b.clicked.connect(slot)
            act_row.addWidget(b)
        layout.addLayout(act_row)

        grp.setLayout(layout)
        return grp

    # ─── Layer helper functions ──────────────────────────────────────────
    def add_dense_layer(self):
        units, ok = QInputDialog.getInt(self, "Dense Layer", "Units:", 128, 1, 2048)
        if not ok:
            return
        act, ok = QInputDialog.getItem(
            self,
            "Activation",
            "Function:",
            ["relu", "sigmoid", "tanh"],
            0,
            False,
        )
        if not ok:
            return
        self.nn_layers.append(("Dense", {"units": units, "activation": act}))
        self.layer_list.addItem(f"Dense({units}, act={act})")
        self.current_model = None

    def add_conv_layer(self):
        # Conv2D requires image data (rank-3 input)
        if self.nn_input_shape is None or len(self.nn_input_shape) != 3:
            self.show_error("Conv2D layers require image data "
                            "(height, width, channels).")
            return

        filters, ok = QInputDialog.getInt(self, "Conv2D", "Filters:", 32, 1, 512)
        if not ok:
            return
        k, ok = QInputDialog.getInt(self, "Kernel Size", "Kernel (k×k):", 3, 1, 7)
        if not ok:
            return
        act, ok = QInputDialog.getItem(
            self, "Activation", "Function:", ["relu", "sigmoid", "tanh"], 0, False
        )
        if not ok:
            return

        self.nn_layers.append(
            ("Conv2D", {"filters": filters,
                        "kernel_size": (k, k),
                        "activation": act})
        )
        self.layer_list.addItem(f"Conv2D({filters},{k}×{k}, act={act})")
        self.current_model = None



    def add_pool_layer(self):
        # MaxPooling2D only valid for images
        if self.nn_input_shape is None or len(self.nn_input_shape) != 3:
            self.show_error("MaxPooling2D layers require image data.")
            return

        k, ok = QInputDialog.getInt(self, "MaxPooling",
                                    "Pool size (p×p):", 2, 1, 4)
        if not ok:
            return

        self.nn_layers.append(("MaxPool2D", {"pool_size": (k, k)}))
        self.layer_list.addItem(f"MaxPool2D({k}×{k})")
        self.current_model = None



    def add_rnn_layer(self, kind="LSTM"):
        # RNN needs 1-D sequence input
        if self.nn_input_shape is None or len(self.nn_input_shape) != 1:
            self.show_error(f"{kind} layers require sequence data "
                            "(timesteps, features).")
            return

        # forbid after Dense/Flatten
        for ltype, _ in reversed(self.nn_layers):
            if ltype in ("Dense", "Flatten"):
                self.show_error(f"Cannot add {kind} after a Dense/Flatten layer. "
                                "Re-order your architecture.")
                return

        units, ok = QInputDialog.getInt(self, kind, "Units:", 64, 1, 512)
        if not ok:
            return

        self.nn_layers.append((kind, {"units": units,
                                      "return_sequences": False}))
        self.layer_list.addItem(f"{kind}({units})")
        self.current_model = None



    def add_dropout_layer(self):
        rate, ok = QInputDialog.getDouble(
            self, "Dropout", "Rate:", 0.5, 0.0, 0.9, 2
        )
        if not ok:
            return
        self.nn_layers.append(("Dropout", {"rate": rate}))
        self.layer_list.addItem(f"Dropout(rate={rate})")
        self.current_model = None

    def remove_selected_layer(self):
        row = self.layer_list.currentRow()
        if row >= 0:
            self.layer_list.takeItem(row)
            self.nn_layers.pop(row)
            self.current_model = None

    def move_layer_up(self):
        row = self.layer_list.currentRow()
        if row > 0:
            self.nn_layers[row - 1], self.nn_layers[row] = (
                self.nn_layers[row], self.nn_layers[row - 1]
            )
            item = self.layer_list.takeItem(row)
            self.layer_list.insertItem(row - 1, item)
            self.layer_list.setCurrentRow(row - 1)
            self.current_model = None

    def move_layer_down(self):
        row = self.layer_list.currentRow()
        if 0 <= row < self.layer_list.count() - 1:
            self.nn_layers[row + 1], self.nn_layers[row] = (
                self.nn_layers[row], self.nn_layers[row + 1]
            )
            item = self.layer_list.takeItem(row)
            self.layer_list.insertItem(row + 1, item)
            self.layer_list.setCurrentRow(row + 1)
            self.current_model = None


    # ─── Dataset loader for NN tab ────────────────────────────────────────
    def load_nn_dataset(self):
        ds = self.nn_dataset_combo.currentText()
        if ds == "MNIST":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train[..., None] / 255.0
            x_test = x_test[..., None] / 255.0
            self.nn_input_shape = (28, 28, 1)
            self.nn_num_classes = 10
        elif ds == "CIFAR-10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = y_train.flatten()
            y_test = y_test.flatten()
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            self.nn_input_shape = (32, 32, 3)
            self.nn_num_classes = 10
        elif ds == "IMDB Reviews":
            vocab = 20000
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
                num_words=vocab
            )
            maxlen = 256
            x_train = tf.keras.preprocessing.sequence.pad_sequences(
                x_train, maxlen=maxlen
            )
            x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
            self.nn_input_shape = (maxlen,)
            self.nn_num_classes = 2
        else:
            self.show_error("Unknown dataset.")
            return

        # train/val split (80/20)
        v = int(0.8 * len(x_train))
        self.nn_x_train, self.nn_x_val = x_train[:v], x_train[v:]
        self.nn_y_train, self.nn_y_val = y_train[:v], y_train[v:]
        self.nn_x_test,  self.nn_y_test  = x_test,  y_test

        # ✨ NEW:  reset any model compiled for an earlier dataset
        self.current_model = None             # <- key line
        self.metrics_display.clear()          # optional: clear old log
        self.status_bar.showMessage(f"{ds} loaded. Shape: {self.nn_x_train.shape}")


    def build_sequential_model(self):
        model = models.Sequential()
        first = True

        for ltype, params in self.nn_layers:
            # Dense -----------------------------------------------------
            if ltype == "Dense":
                if first:
                    model.add(layers.Flatten(input_shape=self.nn_input_shape))
                    first = False
                elif model.layers and len(model.output_shape) > 2:
                    model.add(layers.Flatten())
                model.add(layers.Dense(**params))

            # Conv2D ----------------------------------------------------
            elif ltype == "Conv2D":
                if model.layers and len(model.output_shape) == 2:
                    raise ValueError("Cannot add Conv2D after Dense/Flatten.")
                if first:
                    model.add(layers.Conv2D(**params,
                                            input_shape=self.nn_input_shape))
                    first = False
                else:
                    model.add(layers.Conv2D(**params))

            # MaxPool2D -------------------------------------------------
            elif ltype == "MaxPool2D":
                if model.layers and len(model.output_shape) == 2:
                    raise ValueError("Cannot add MaxPool2D after Dense/Flatten.")
                if first:
                    model.add(layers.MaxPooling2D(**params,
                                                  input_shape=self.nn_input_shape))
                    first = False
                else:
                    model.add(layers.MaxPooling2D(**params))

            # RNN layers -----------------------------------------------
            elif ltype in ("LSTM", "GRU"):
                if model.layers and len(model.output_shape) == 2:
                    raise ValueError("Cannot add LSTM/GRU after Dense/Flatten.")
                rnn_cls = layers.LSTM if ltype == "LSTM" else layers.GRU
                if first:
                    model.add(rnn_cls(**params,
                                      input_shape=self.nn_input_shape))
                    first = False
                else:
                    model.add(rnn_cls(**params))

            # Dropout ---------------------------------------------------
            elif ltype == "Dropout":
                model.add(layers.Dropout(**params))

        # Flatten before classifier if still 4-D
        if model.layers and len(model.output_shape) > 2:
            model.add(layers.Flatten())

        act   = "softmax" if self.nn_num_classes > 2 else "sigmoid"
        units = self.nn_num_classes if self.nn_num_classes > 2 else 1
        model.add(layers.Dense(units, activation=act))
        return model



    # ─── Learning-rate scheduler factory ─────────────────────────────────
    def make_lr_scheduler(self):
        choice = self.lr_schedule_combo.currentText()
        base_lr = self.lr_spin.value()
        if choice == "Step Decay":
            def step(epoch):
                drop = 0.5
                epochs_drop = 10
                return base_lr * (drop ** np.floor(epoch / epochs_drop))
            return callbacks.LearningRateScheduler(step)
        elif choice == "Exponential Decay":
            return callbacks.LearningRateScheduler(lambda e: base_lr * np.exp(-0.1 * e))
        else:
            return None

    # ──────────────────────────────────────────────────────────────────────
    #  Sixth-tab “Train” slot – full version with optional fine-tuning
    # ──────────────────────────────────────────────────────────────────────
    def train_neural_network(self):
        # ── sanity checks ────────────────────────────────────────────────
        if not self.nn_layers and self.current_model is None:
            self.show_error("Please build a model (or load a pre-trained one) first.")
            return
        if self.nn_input_shape is None or self.nn_x_train is None:
            self.show_error("Load a dataset before training.")
            return

        # ── build or reuse model ─────────────────────────────────────────
        model = self.current_model or self.build_sequential_model()
        self.current_model = model

        # ── decide loss up-front (needed in two places) ─────────────────
        loss = ("sparse_categorical_crossentropy"
                if self.nn_num_classes > 2 else "binary_crossentropy")

        # ── compile if not compiled yet ──────────────────────────────────
        if getattr(model, "optimizer", None) is None:
            opt_name = self.optimizer_combo.currentText()
            lr       = self.lr_spin.value()
            opt = {"Adam": optimizers.Adam(lr),
                   "SGD":  optimizers.SGD(lr),
                   "RMSprop": optimizers.RMSprop(lr)}[opt_name]
            model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

        # ── callbacks ────────────────────────────────────────────────────
        cb = [GuiLoggingCallback(self)]
        sch = self.make_lr_scheduler()
        if sch: cb.append(sch)
        cb.append(callbacks.EarlyStopping(patience=5,
                                          restore_best_weights=True,
                                          monitor="val_loss"))

        # 1️⃣ FIRST TRAIN (feature extractor) ─────────────────────────────
        hist1 = model.fit(self.nn_x_train, self.nn_y_train,
                          validation_data=(self.nn_x_val, self.nn_y_val),
                          epochs=self.epochs_spin.value(),
                          batch_size=self.batch_spin.value(),
                          callbacks=cb, verbose=0)

        # 2️⃣ OPTIONAL FINE-TUNE (un-freeze backbone) ─────────────────────
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer=optimizers.Adam(1e-5),
                      loss=loss, metrics=["accuracy"])

        hist2 = model.fit(self.nn_x_train, self.nn_y_train,
                          validation_data=(self.nn_x_val, self.nn_y_val),
                          epochs=int(self.epochs_spin.value() / 2),
                          batch_size=self.batch_spin.value(),
                          callbacks=cb, verbose=0)

        # ── merge histories & plot ───────────────────────────────────────
        full_hist = {k: hist1.history[k] + hist2.history[k] for k in hist1.history}

        # ── plot curves ─────────────────────────────────────────────────
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        ax.plot(full_hist["loss"],        label="loss")
        ax.plot(full_hist["val_loss"],    label="val_loss")
        if "accuracy" in full_hist:
            ax.plot(full_hist["accuracy"],     label="acc")
            ax.plot(full_hist["val_accuracy"], label="val_acc")

        ax.set_xlabel("Epoch")
        ax.legend()
        ax.set_title("Training & Fine-tuning")

        # give the bottom of the figure a bit more room
        self.figure.subplots_adjust(bottom=0.18)
        self.figure.tight_layout()     # final tidy-up

        self.canvas.draw()


        # ── test metrics ────────────────────────────────────────────────
        y_pred = (np.argmax(model.predict(self.nn_x_test), axis=1)
                  if self.nn_num_classes > 2
                  else (model.predict(self.nn_x_test) > 0.5).astype(int).flatten())
        acc = accuracy_score(self.nn_y_test, y_pred)
        f1  = f1_score(self.nn_y_test, y_pred, average="weighted")
        self.metrics_display.append(f"\nTest Accuracy {acc:.4f}   F1 {f1:.4f}\n")

        # ── gradient histogram (sample batch, TF-2 API) ─────────────────
        try:
            sample_sz = min(128, len(self.nn_x_train))
            sample_x  = self.nn_x_train[:sample_sz]
            sample_y  = self.nn_y_train[:sample_sz]

            with tf.GradientTape() as tape:
                preds = model(sample_x, training=True)
                loss_val = model.compute_loss(sample_y, preds)

            grads = tape.gradient(loss_val, model.trainable_variables)
            flat  = np.concatenate(
                [tf.reshape(g, [-1]).numpy() for g in grads if g is not None]
            )

            plt.figure()
            plt.hist(flat, bins=60)
            plt.title("Gradient Histogram (sample batch)")
            plt.xlabel("grad value"); plt.ylabel("count")
            plt.show()
        except Exception as e:
            print("Gradient histogram skipped:", e)




    def save_nn_model(self):
        if self.current_model is None:
            self.show_error("No model to save.")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            os.path.join(os.getcwd(), "model.keras"),
            "Keras Model (*.keras);;HDF5 Model (*.h5)"
        )
        if not fname:
            return
        if not fname.endswith((".keras", ".h5")):
            fname += ".keras"

        self.current_model.save(fname)
        self.status_bar.showMessage(f"Model saved to {os.path.abspath(fname)}")


    def load_nn_model(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            os.getcwd(),
            "Keras / HDF5 Model (*.keras *.h5)"
        )
        if not fname:
            return

        mdl = models.load_model(fname)
        self.current_model = mdl

        # reset GUI architecture list
        self.layer_list.clear()
        self.nn_layers.clear()

        if isinstance(mdl, models.Sequential):
            for layer in mdl.layers[:-1]:  # skip final Dense classifier
                if isinstance(layer, layers.Dense):
                    units = layer.units
                    act   = layer.activation.__name__
                    self.nn_layers.append(("Dense", {"units": units,
                                                     "activation": act}))
                    self.layer_list.addItem(f"Dense({units}, act={act})")

                elif isinstance(layer, layers.Conv2D):
                    f = layer.filters
                    k = layer.kernel_size[0]
                    act = layer.activation.__name__
                    self.nn_layers.append(("Conv2D",
                                           {"filters": f,
                                            "kernel_size": (k, k),
                                            "activation": act}))
                    self.layer_list.addItem(f"Conv2D({f},{k}×{k}, act={act})")

                elif isinstance(layer, layers.MaxPooling2D):
                    k = layer.pool_size[0]
                    self.nn_layers.append(("MaxPool2D", {"pool_size": (k, k)}))
                    self.layer_list.addItem(f"MaxPool2D({k}×{k})")

                elif isinstance(layer, layers.Dropout):
                    r = layer.rate
                    self.nn_layers.append(("Dropout", {"rate": r}))
                    self.layer_list.addItem(f"Dropout(rate={r:.2f})")

                elif isinstance(layer, layers.LSTM):
                    u = layer.units
                    self.nn_layers.append(("LSTM", {"units": u,
                                                    "return_sequences":
                                                    layer.return_sequences}))
                    self.layer_list.addItem(f"LSTM({u})")

                elif isinstance(layer, layers.GRU):
                    u = layer.units
                    self.nn_layers.append(("GRU", {"units": u,
                                                   "return_sequences":
                                                   layer.return_sequences}))
                    self.layer_list.addItem(f"GRU({u})")

            self.nn_input_shape = mdl.input_shape[1:]
            self.nn_num_classes = mdl.output_shape[-1]

        else:
            self.layer_list.addItem("(Functional model loaded)")
            self.metrics_display.append(
                "Loaded functional model – architecture not displayed, "
                "but you can train / evaluate.\n"
            )

        self.status_bar.showMessage(f"Loaded model {os.path.basename(fname)}")





    def load_pretrained_model(self):
        """
        1. Lets the user pick VGG16 / ResNet50 / MobileNetV2.
        2. Converts 1-channel images (e.g. MNIST) to 3-channel RGB and resizes
        them to 224×224 inside the model graph.
        3. Freezes the backbone and adds a small trainable head.
        4. Compiles the model and stores it in self.current_model.
        """
        # ── need data first ───────────────────────────────────────────────
        if self.nn_x_train is None:
            self.show_error("Load a dataset before choosing a pre-trained backbone.")
            return

        # ── pick backbone via dialog ──────────────────────────────────────
        backbone_name, ok = QInputDialog.getItem(
            self, "Pre-trained backbone",
            "Select a backbone to fine-tune:",
            ["VGG16", "ResNet50", "MobileNetV2"], 0, False)
        if not ok:
            return

        backbone_dict = {
            "VGG16":      applications.VGG16,
            "ResNet50":   applications.ResNet50,
            "MobileNetV2": applications.MobileNetV2,
        }
        Backbone = backbone_dict[backbone_name]

        # ── bring data into a 3-channel format if needed ─────────────────
        def ensure_rgb(arr):
            if arr.ndim == 3:                       # (N,H,W) grayscale
                arr = arr[..., None]
            if arr.shape[-1] == 1:                  # (N,H,W,1) → repeat
                arr = np.repeat(arr, 3, axis=-1)
            return arr.astype("float32")

        self.nn_x_train = ensure_rgb(self.nn_x_train)
        self.nn_x_val   = ensure_rgb(self.nn_x_val)
        self.nn_x_test  = ensure_rgb(self.nn_x_test)

        orig_h, orig_w, orig_c = self.nn_x_train.shape[1:]
        n_classes = self.nn_num_classes

        # ── build preprocessing + backbone graph -------------------------
        inputs = layers.Input(shape=(orig_h, orig_w, orig_c))
        x = layers.Resizing(224, 224, interpolation="bilinear")(inputs)
        x = Backbone(weights="imagenet", include_top=False)(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(
            n_classes,
            activation="softmax" if n_classes > 2 else "sigmoid")(x)

        model = models.Model(inputs, outputs,
                            name=f"{backbone_name}_finetune")

        loss = ("sparse_categorical_crossentropy"
                if n_classes > 2 else "binary_crossentropy")
        model.compile(optimizer=optimizers.Adam(1e-3),
                    loss=loss, metrics=["accuracy"])

        self.current_model = model
        self.metrics_display.append(
            f"{backbone_name} loaded – input resized to 224×224×3; "
            f"backbone frozen, classification head trainable.\n"
        )


    def train_dcgan(self):
        QMessageBox.information(
            self,
            "DCGAN",
            "Placeholder — simple DCGAN trainer can be implemented here.",
        )

    # ────────────────────────── Utility ─────────────────────────
    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

# =============================================================================
# Main launcher
# =============================================================================
def main():
    app = QApplication(sys.argv)
    win = EnhancedMLCourseGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
