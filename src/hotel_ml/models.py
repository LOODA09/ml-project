from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import CONFIG

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    Sequential = None
    Conv1D = Dense = Dropout = Flatten = Input = MaxPooling1D = None
    Adam = None
    TENSORFLOW_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


@dataclass
class ModelSpec:
    name: str
    estimator: BaseEstimator
    complexity: str


class KerasCNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs: int = 20, batch_size: int = 32, verbose: int = 0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def _build_model(self, input_length: int):
        model = Sequential(
            [
                Input(shape=(input_length, 1)),
                Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
                Flatten(),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, x, y):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed; cannot train 1D-CNN.")

        tf.keras.utils.set_random_seed(CONFIG.random_state)
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        x_array = x_array.reshape((x_array.shape[0], x_array.shape[1], 1))
        self.model_ = self._build_model(x_array.shape[1])
        classes = np.unique(y_array.astype(int))
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_array.astype(int),
        )
        class_weight_map = {
            int(label): float(weight) for label, weight in zip(classes, class_weights)
        }
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ]
        self.model_.fit(
            x_array,
            y_array,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.1,
            callbacks=callbacks,
            class_weight=class_weight_map,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        x_array = np.asarray(x, dtype=np.float32)
        x_array = x_array.reshape((x_array.shape[0], x_array.shape[1], 1))
        probs = self.model_.predict(x_array, verbose=0).reshape(-1)
        return np.column_stack([1 - probs, probs])

    def predict(self, x):
        probs = self.predict_proba(x)[:, 1]
        return (probs >= 0.5).astype(int)

    def score(self, x, y):
        return accuracy_score(y, self.predict(x))


def get_model_specs(include_svm: bool = False) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="Naive Bayes",
            estimator=GaussianNB(var_smoothing=1e-8),
            complexity="Low",
        ),
        ModelSpec(
            name="Logistic Regression",
            estimator=LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                random_state=CONFIG.random_state,
            ),
            complexity="Low",
        ),
        ModelSpec(
            name="KNN",
            estimator=KNeighborsClassifier(n_neighbors=11, weights="distance"),
            complexity="Medium",
        ),
        ModelSpec(
            name="Decision Tree",
            estimator=DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=4,
                min_samples_split=20,
                class_weight="balanced",
                random_state=CONFIG.random_state,
            ),
            complexity="Low",
        ),
        ModelSpec(
            name="Random Forest",
            estimator=RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                min_samples_split=10,
                class_weight="balanced_subsample",
                random_state=CONFIG.random_state,
                n_jobs=-1,
            ),
            complexity="Medium",
        ),
    ]

    if XGBOOST_AVAILABLE:
        specs.append(
            ModelSpec(
                name="XGBoost",
                estimator=XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    reg_alpha=0.0,
                    random_state=CONFIG.random_state,
                    n_jobs=-1,
                ),
                complexity="High",
            )
        )

    specs.extend(
        [
        ModelSpec(
            name="MLP",
            estimator=MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                alpha=0.0005,
                learning_rate_init=0.001,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                max_iter=500,
                random_state=CONFIG.random_state,
            ),
            complexity="High",
        ),
        ]
    )

    if include_svm:
        specs.append(
            ModelSpec(
                name="SVM",
                estimator=SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=CONFIG.random_state,
                ),
                complexity="High",
            )
        )

    if TENSORFLOW_AVAILABLE:
        specs.append(
            ModelSpec(
                name="1D-CNN",
                estimator=KerasCNNClassifier(epochs=20, batch_size=64, verbose=0),
                complexity="High",
            )
        )

    return specs
