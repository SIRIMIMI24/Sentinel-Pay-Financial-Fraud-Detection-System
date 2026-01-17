import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Mocking internal imports for the example
from src.logger import get_logger
from src.custom_exception import CustomException

class DataProcessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        # Unified preprocessor replaces individual encoders and scalers
        self.preprocessor = None 
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []

        os.makedirs(self.output_path, exist_ok=True)
        print("DataProcessing initialized....")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            print("Data loaded successfully...")
        except Exception as e:
            raise Exception(f"Failed to load data: {e}")

    def preprocess_data(self):
        """Clean raw data and generate derived features."""
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
            
            print("Data preprocessing (cleaning) completed.")
        except Exception as e:
            raise Exception(f"Preprocessing failed: {e}")

    def feature_selection(self):
        """Select top 10 features and identify their types for the preprocessor."""
        try:
            # Temporary encoding for Mutual Info calculation
            X_tmp = self.X.copy()
            for col in X_tmp.select_dtypes(include=['object', 'category']).columns:
                X_tmp[col] = X_tmp[col].astype('category').cat.codes

            selector = SelectKBest(
                score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), 
                k=10
            )
            selector.fit(X_tmp, self.y)
            
            self.selected_features = self.X.columns[selector.get_support()].tolist()
            # Filter X to only the 10 MI features
            self.X = self.X[self.selected_features]
            
            print(f"Top 10 MI features: {self.selected_features}")
        except Exception as e:
            raise Exception(f"Feature selection failed: {e}")

    def split_and_build_pipeline(self):
        """Construct the ColumnTransformer and transform data."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )

            # Identify column types automatically from selected features
            cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()

            # Build unified preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
                ]
            )

            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)

            return X_train_transformed, X_test_transformed, y_train, y_test
        except Exception as e:
            raise Exception(f"Pipeline construction failed: {e}")

    def save_artifacts(self, X_train, X_test, y_train, y_test):
        """Save the processed data and the unified preprocessor pkl."""
        try:
            joblib.dump(X_train, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))

            # Save the preprocessor that Flask will use
            joblib.dump(self.preprocessor, os.path.join(self.output_path, "preprocessor.pkl"))
            print("All artifacts saved successfully.")
        except Exception as e:
            raise Exception(f"Saving artifacts failed: {e}")

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.feature_selection()
        X_train, X_test, y_train, y_test = self.split_and_build_pipeline()
        self.save_artifacts(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    input_path = "artifacts/raw/data_financial_detection.csv"
    output_path = "artifacts/processed"
    processor = DataProcessing(input_path, output_path)
    processor.run()