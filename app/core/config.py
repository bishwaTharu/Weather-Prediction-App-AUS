from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    PROJECT_NAME: str = "Weather Prediction API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    ASSETS_DIR: Path = BASE_DIR / "assets"
    
    PREPROCESSOR_PATH: Path = ASSETS_DIR / "preprocessor.joblib"
    MODEL_PATH: Path = ASSETS_DIR / "best_model.joblib"

    class Config:
        env_file = ".env"

settings = Settings()
