import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    """
    Handles model training and evaluation with 
    robustness against feature name warnings and metric undefined states.
    """
    def __init__(self, processed_data_path: str = "artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/model"
        os.makedirs(self.model_dir, exist_ok=True)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        logger.info("ModelTraining initialized.")

    def load_data(self):
        try:
            # Load as DataFrames to preserve feature names
            self.X_train = joblib.load(os.path.join(self.processed_data_path, 'X_train.pkl'))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, 'X_test.pkl'))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, 'y_train.pkl'))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, 'y_test.pkl'))

            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise

    def train_model(self):
        try:
            # Added scale_pos_weight to handle potential imbalance in financial data
            self.model = LGBMClassifier(
                objective='binary',
                n_estimators=300,
                learning_rate=0.01,
                max_depth=4,
                min_child_samples=5,
                random_state=42,
                importance_type='gain',
                verbose=-1
            )
            self.model.fit(self.X_train, self.y_train)

            joblib.dump(self.model, os.path.join(self.model_dir, 'model.pkl'))
            logger.info("Model trained and saved.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self):
        try:
            # Ensure X_test is a DataFrame to silence UserWarning
            if not isinstance(self.X_test, pd.DataFrame):
                self.X_test = pd.DataFrame(self.X_test)

            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1]

            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Using zero_division=0 to silence UndefinedMetricWarning
            precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_proba)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)


            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": roc_auc
            }
            
            logger.info(f"Evaluation Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    with mlflow.start_run():
        trainer = ModelTraining()
        trainer.run()

