from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class WeatherPredictionRequest(BaseModel):
    # Defining core weather features based on standard Australian weather datasets
    # Making them optional to handle partial data, but providing defaults or nulls
    MinTemp: Optional[float] = None
    MaxTemp: Optional[float] = None
    Rainfall: Optional[float] = 0.0
    Evaporation: Optional[float] = None
    Sunshine: Optional[float] = None
    WindGustDir: Optional[str] = None
    WindGustSpeed: Optional[float] = None
    WindDir9am: Optional[str] = None
    WindDir3pm: Optional[str] = None
    WindSpeed9am: Optional[float] = None
    WindSpeed3pm: Optional[float] = None
    Humidity9am: Optional[float] = None
    Humidity3pm: Optional[float] = None
    Pressure9am: Optional[float] = None
    Pressure3pm: Optional[float] = None
    Cloud9am: Optional[float] = None
    Cloud3pm: Optional[float] = None
    Temp9am: Optional[float] = None
    Temp3pm: Optional[float] = None
    RainToday: Optional[str] = "No"

    class Config:
        json_schema_extra = {
            "example": {
                "MinTemp": 13.4,
                "MaxTemp": 22.9,
                "Rainfall": 0.6,
                "WindGustSpeed": 44.0,
                "Humidity9am": 71.0,
                "Humidity3pm": 22.0,
                "Pressure9am": 1007.7,
                "Pressure3pm": 1007.1,
                "RainToday": "No"
            }
        }

class WeatherPredictionResponse(BaseModel):
    prediction: str
    probability: float
    status: str = "success"
