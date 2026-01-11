import os
import kagglehub
import shutil
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import * # Calls DATASET_NAME , TARGET_DIR
import zipfile
import json
import pandas as pd

# Before run Clen Cache: pip install kagglehub

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, dataset_name:str, target_dir:str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

        # Select data ingestion
        self.required_files =[
            "cards_data.csv",
            "mcc_codes.json",
            "train_fraud_labels.json",
            "transactions_data.csv",
            "users_data.csv",
        ]

    def create_raw_dir(self):
        raw_dir = os.path.join(self.target_dir, "raw")
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir)
                logger.info(f"Created the {raw_dir}")
            except Exception as e:
                logger.error("Erro while creating directory..")
                raise CustomException("Faile to create raw dir", e)
        return raw_dir
    
    def extract_zip(self, path:str) -> str:
        
        if os.path.isdir(path):
            # kagglehub downloads as folder if dataset is small
            logger.info(f"Dataset is already extracted: {path}")
            return path

        if path.endswith(".zip"):
            logger.info(f"Extracting zip file: {path}")
            extract_dir = path.replace(".zip", "")
            try:
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                logger.info(f"Extracted zip file to: {extract_dir}")
                return extract_dir
            except Exception as e:
                logger.error("Error while extracting zip file.")
                raise CustomException("Failed to extract zip file", e)
            
        
        raise CustomException("Provided path is neither a directory nor a zip file.")
        
    def extract_dataset_files(self, dataset_root: str, raw_dir: str) -> None:
        try:
            for fname in self.required_files:
                src = os.path.join(dataset_root, fname)
                dst = os.path.join(raw_dir, fname)

                if os.path.exists(src):
                    shutil.copy(src, dst)
                    logger.info(f"Move {src} to {dst}")
                else:
                    logger.warning(f"Required file {fname} not found in dataset.")
        except Exception as e:
            logger.error("Error while extracting required dataset files.")
            raise CustomException("Failed to extract dataset files", e)
    
    def download_data(self, raw_dir: str) -> None:
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Downloaded dataset to: {path}")

            dataset_root = self.extract_zip(path)
            self.extract_dataset_files(dataset_root, raw_dir)
        
        except Exception as e:
            logger.error("Error while downloading data.")
            raise CustomException("Failed to download data", e)
        
    def combine_and_save_raw(self, raw_dir: str) -> str:
        """
        Load raw files, merge them into a modeling-ready dataframe,
        and save as data_financial_detection.csv
        """
        try:
            logger.info("Loading raw CSV and JSON files...")

            # 1) Load files
            df_cards = pd.read_csv(os.path.join(raw_dir, "cards_data.csv"))
            df_transactions = pd.read_csv(os.path.join(raw_dir, "transactions_data.csv"))
            df_user_data = pd.read_csv(os.path.join(raw_dir, "users_data.csv"))
            df_mcc = pd.read_json(os.path.join(raw_dir, "mcc_codes.json"), orient="index")

            # 2) Clean MCC table
            df_mcc.reset_index(inplace=True)
            df_mcc.columns = ["mcc_code", "description"]

            # 3) Fraud labels
            with open(os.path.join(raw_dir, "train_fraud_labels.json"), "r") as f:
                fraud_labels = json.load(f)["target"]

            df_fraud = (
                pd.DataFrame(list(fraud_labels.items()), columns=["transaction_id", "is_fraud"])
                .assign(is_fraud=lambda d: d["is_fraud"].map({"No": 0, "Yes": 1}))
            )
            df_fraud["transaction_id"] = df_fraud["transaction_id"].astype(int)

            # 4) Merge tables
            logger.info("Joining datasets...")

            df = df_transactions.merge(df_fraud, left_on="id", right_on="transaction_id", how="inner")
            df = df.merge(df_cards, left_on="card_id", right_on="id", how="left", suffixes=("", "_card"))
            df = df.merge(df_user_data, left_on="client_id", right_on="id", how="left", suffixes=("", "_user"))
            df = df.merge(df_mcc, left_on="mcc", right_on="mcc_code", how="left")

            # Optional drop columns
            drop_cols = [
                "mcc", "id_card", "client_id_card", "card_number", "cvv",
                "acc_open_date", "id_user", "address"
            ]
            df.drop(columns=drop_cols, inplace=True, errors="ignore")

            # 5) Save output
            output_path = os.path.join(raw_dir, "data_financial_detection.csv")
            df.to_csv(output_path, index=False)

            logger.info(f"Final merged file saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error("Error while combining and saving raw dataset.")
            raise CustomException("Failed combining raw dataset", e)

    def initiate_data_ingestion(self) -> None:
        try:
            raw_dir = self.create_raw_dir()
            self.download_data(raw_dir)
            self.combine_and_save_raw(raw_dir)
            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error("Error during data ingestion process.")
            raise CustomException("Data ingestion failed", e)
    
if __name__ == "__main__":
    data_ingestion = DataIngestion(
        dataset_name=DATASET_NAME,
        target_dir=TARGET_DIR
    )
    data_ingestion.initiate_data_ingestion()
    