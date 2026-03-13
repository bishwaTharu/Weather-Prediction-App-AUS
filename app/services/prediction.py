import pandas as pd
import logging
import joblib
from typing import Any
from app.core.config import settings
from app.models.schemas import WeatherPredictionRequest
from app.services.preprocessor import WeatherDataPreprocessor

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_artifacts()

    def load_artifacts(self):
        try:
            if settings.MODEL_PATH.exists() and settings.PREPROCESSOR_PATH.exists():
                logger.info(f"Loading model from {settings.MODEL_PATH}")
                self.model = joblib.load(settings.MODEL_PATH)
                logger.info(f"Loading preprocessor from {settings.PREPROCESSOR_PATH}")
                self.preprocessor = WeatherDataPreprocessor.load(settings.PREPROCESSOR_PATH)
            else:
                logger.warning("Artifacts not found. Prediction will fail.")
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")

    def predict(self, data: WeatherPredictionRequest) -> Any:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or Preprocessor not loaded!")

        df = pd.DataFrame([data.dict()])
        X_processed = self.preprocessor.transform(df)
        prediction = self.model.predict(X_processed)
        probability = 0.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_processed)
            probability = float(max(probs[0]))

        return {
            "prediction": "Yes" if prediction[0] == 1 else "No",
            "probability": probability
        }

prediction_service = PredictionService()
