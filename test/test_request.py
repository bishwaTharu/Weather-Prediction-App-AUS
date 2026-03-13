import requests
import json

def test_prediction():
    url = "http://localhost:8000/api/v1/predict"
    
    # Sample data for prediction
    payload = {
        "MinTemp": 15.0,
        "MaxTemp": 25.0,
        "Rainfall": 0.0,
        "Evaporation": 5.0,
        "Sunshine": 10.0,
        "WindGustDir": "W",
        "WindGustSpeed": 35.0,
        "WindDir9am": "W",
        "WindDir3pm": "WNW",
        "WindSpeed9am": 15.0,
        "WindSpeed3pm": 20.0,
        "Humidity9am": 50.0,
        "Humidity3pm": 40.0,
        "Pressure9am": 1015.0,
        "Pressure3pm": 1012.0,
        "Cloud9am": 1.0,
        "Cloud3pm": 1.0,
        "Temp9am": 18.0,
        "Temp3pm": 23.0,
        "RainToday": "No"
    }
    
    print(f"Sending prediction request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running at http://localhost:8000?")
    except Exception as e:
        print(f"An error occurred: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Detail: {e.response.text}")

if __name__ == "__main__":
    test_prediction()
