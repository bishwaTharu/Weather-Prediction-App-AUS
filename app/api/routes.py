from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import WeatherPredictionRequest, WeatherPredictionResponse
from app.services.prediction import PredictionService
from app.api.dependencies import get_prediction_service

router = APIRouter()

@router.post("/predict", response_model=WeatherPredictionResponse)
async def predict(
    request: WeatherPredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
):
    try:
        results = service.predict(request)
        return WeatherPredictionResponse(**results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
