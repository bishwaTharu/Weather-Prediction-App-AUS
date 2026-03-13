import pandas as pd
import numpy as np
import logging
import os
from app.services.preprocessor import WeatherDataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_cleaning():
    data_dir = "data"
    assets_dir = "assets"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    train_file = os.path.join(data_dir, "Weather Training Data.csv")
    test_file = os.path.join(data_dir, "Weather Test Data.csv")
    
    train_out = os.path.join(data_dir, "cleaned_Weather_Training_Data.csv")
    test_out = os.path.join(data_dir, "cleaned_Weather_Test_Data.csv")
    
    if not os.path.exists(train_file):
        logger.error(f"Training file {train_file} not found. Cannot proceed.")
        return

    logger.info("--- CLEANING TRAINING DATA ---")
    df_train = pd.read_csv(train_file)
    preprocessor = WeatherDataPreprocessor()
    
    df_train_cleaned = preprocessor.fit_transform(df_train)
    df_train_cleaned.to_csv(train_out, index=False)
    logger.info(f"Saved cleaned training data to {train_out}")
    
    # Save the artifacts!
    preprocessor.save(assets_dir)
    
    # Process test data if exists
    if os.path.exists(test_file):
        df_test = pd.read_csv(test_file)
        df_test_cleaned = preprocessor.transform(df_test)
        df_test_cleaned.to_csv(test_out, index=False)
        logger.info(f"Saved cleaned test data to {test_out}")
    else:
        logger.warning(f"Test file {test_file} not found. Skipping.")

if __name__ == "__main__":
    run_data_cleaning()
