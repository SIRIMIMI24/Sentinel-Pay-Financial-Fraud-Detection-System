import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []

        os.makedirs(self.output_path, exist_ok=True)
        logger.info("DataProcessing initialized....")   

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data loaded sucesfully...")
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data")
    
    def preprocess_data(self):
        try:
            # 1. Handle In-place operations correctly
            self.df.drop_duplicates(inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            # 2. Vectorized Missing Value Imputation
            fill_values = {
                "merchant_state": "UNKNOWN",
                "zip": "00000",
                "errors": "is_not_error"
            }
            self.df.fillna(value=fill_values, inplace=True)

            # 3. Filtering Outliers
            self.df = self.df[self.df["current_age"] <= 100].copy()

            # 4. Optimized Money Conversion (Vectorized)
            money_cols = ["amount", "per_capita_income", "yearly_income", "total_debt", "credit_limit"]
            for col in money_cols:
                if col in self.df.columns:
                    self.df[col] = (
                        self.df[col]
                        .astype(str)
                        .str.replace(r"[\$,]", "", regex=True)
                        .str.strip()
                    )
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

            # 5. Temporal Feature Engineering
            date_cols = ["date", "acct_open_date", "expires"]
            for col in date_cols:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

            self.df = self.df[self.df["date"] >= "2018-06-01"].copy()

            # Time-based derivations
            self.df["txn_hour"] = self.df["date"].dt.hour
            self.df["txn_dayofweek"] = self.df["date"].dt.dayofweek
            self.df["account_tenure_days"] = (self.df["date"] - self.df["acct_open_date"]).dt.days
            self.df["days_to_expire"] = (self.df["expires"] - self.df["date"]).dt.days
            
            # Fixed logic for year-based subtraction
            if "year_pin_last_changed" in self.df.columns:
                self.df["days_since_pin_change"] = self.df["date"].dt.year - self.df["year_pin_last_changed"]

            # 6. Financial Ratios (Using epsilon for stability)
            eps = 1e-6
            self.df["card_utilization"] = self.df["amount"] / (self.df["credit_limit"] + eps)
            self.df["income_to_debt_ratio"] = self.df["yearly_income"] / (self.df["total_debt"] + eps)
            self.df["has_errors"] = (self.df["errors"] != "is_not_error").astype(int)

            # 7. Discretization
            self.df["age_bucket"] = pd.cut(
                self.df["current_age"],
                bins=[0, 25, 40, 60, 100], # Removed duplicate 60
                labels=["<25", "25-40", "40-60", "60-100"]
            )

            # 8. Feature Selection & Target Separation
            drop_cols = [
                "id", "transaction_id", "client_id", "card_id", "merchant_id",
                "date", "expires", "year_pin_last_changed", "acct_open_date", 
                "errors", "birth_year", "birth_month", "retirement_age", 
                "per_capita_income", "num_credit_card"
            ]
            self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], inplace=True)

            self.X = self.df.drop(columns=['is_fraud'])
            self.y = self.df['is_fraud']

            # 9. Encoding (Categorical to Numerical)
            # Note: Ensure age_bucket is handled as it is now a 'category' type
            cat_features = self.X.select_dtypes(include=['object', 'category']).columns
            for col in cat_features:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le

            logger.info("Data preprocessing completed successfully.")

        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            raise CustomException("Data preprocessing failed")

    def feature_selection(self):
        try:
            # 1. Prevent Leakage: Use a temporary split for ranking
            X_train_temp, _, y_train_temp, _ = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )

            # 2. Use Mutual Information (Handles continuous data + non-linearities)
            # discrete_features: Identify indices of categorical columns for MI
            discrete_mask = (X_train_temp.dtypes == 'int64') 
            
            selector = SelectKBest(
                score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), 
                k=10
            )
            
            selector.fit(X_train_temp, y_train_temp)
            
            # 3. Persist feature names rather than modifying self.X immediately
            self.selected_features = X_train_temp.columns[selector.get_support()].tolist()
            
            # 4. Map back to the main dataset
            self.X = self.X[self.selected_features]
            
            logger.info(f"Top 10 features selected via Mutual Info: {self.selected_features}")

        except Exception as e:
            logger.error(f"Feature Selection Error: {str(e)}")
            raise e
    
    def split_and_scale_data(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            logger.info("Data splitting and scaling completed...")

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error during data splitting and scaling: {e}")
            raise CustomException("Failed to split and scale data")
    
    def save_data_scaler(self, X_train, X_test, y_train, y_test):
        try:
            joblib.dump(X_train, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))

            joblib.dump(self.scaler, os.path.join(self.output_path, "scaler.pkl"))

            logger.info("Data and scaler saved successfully...")
        
        except Exception as e:
            logger.error(f"Error while saving data/scaler: {e}")
            raise CustomException("Failed to save data/scaler")
    
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.feature_selection()
        X_train, X_test, y_train, y_test = self.split_and_scale_data()
        self.save_data_scaler(X_train, X_test, y_train, y_test)

        logger.info("Data processing pipelinecompleted successfully...")

if __name__ == "__main__":
    input_path = "artifacts/raw/data_financial_detection.csv"
    output_path = "artifacts/processed"

    processor = DataProcessing(input_path, output_path)
    processor.run()
