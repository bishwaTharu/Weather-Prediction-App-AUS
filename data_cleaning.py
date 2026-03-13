import pandas as pd
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
    train_out = os.path.join(data_dir, "cleaned_Weather_Training_Data.csv")
    
    if not os.path.exists(train_file):
        logger.error(f"Training file {train_file} not found. Cannot proceed.")
        return

    logger.info("--- CLEANING AND BALANCING TRAINING DATA ---")
    df_train = pd.read_csv(train_file)
    
    # Check for imbalance before cleaning (to potentially avoid cleaning redundant data or to ensure we have enough samples)
    target_col = 'RainTomorrow'
    if target_col in df_train.columns:
        counts = df_train[target_col].value_counts()
        logger.info(f"Class distribution before balancing:\n{counts}")
        
        # Simple heuristic: if minority class is < 80% of majority, let's balance
        if counts.min() / counts.max() < 0.8:
            logger.info("Imbalance detected. Applying random oversampling...")
            class_max = counts.idxmax()
            class_min = counts.idxmin()
            
            df_minority = df_train[df_train[target_col] == class_min]
            df_majority = df_train[df_train[target_col] == class_max]
            
            # Oversample minority to match majority
            df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=42)
            df_train = pd.concat([df_majority, df_minority_upsampled])
            
            logger.info(f"New class distribution:\n{df_train[target_col].value_counts()}")
        else:
            logger.info("Dataset is balanced enough. No oversampling needed.")
    
    preprocessor = WeatherDataPreprocessor()
    df_train_cleaned = preprocessor.fit_transform(df_train)
    df_train_cleaned.to_csv(train_out, index=False)
    logger.info(f"Saved cleaned and balanced training data to {train_out}")
    
    # Save the artifacts!
    preprocessor.save(assets_dir)
    
    logger.info("Data preparation completed. Skipping test data cleaning as per requirement.")

if __name__ == "__main__":
    run_data_cleaning()
