from app.services.prediction import prediction_service

def get_prediction_service():
    # Load once and yield (FastAPI dependency)
    return prediction_service