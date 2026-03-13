# Weather Prediction API 🌦️

A production-ready FastAPI application that hosts a Machine Learning model to predict whether it will rain tomorrow in Australia. This project demonstrates a clean separation of concerns between model training, data preprocessing, and API serving.

## 🚀 Features

- **Production-Ready FastAPI**: Structured directory layout with asynchronous endpoints.
- **Robust Preprocessing**: Custom `WeatherDataPreprocessor` that prevents data leakage by persisting imputation and scaling statistics from training.
- **Model Persistence**: Automatically saves and loads the best-performing model (SVM/Decision Tree) using `joblib`.
- **Probability Estimation**: Returns non-null confidence scores for every prediction.
- **Modern Dependency Management**: Built and managed with `uv` for speed and reproducibility.


## 🛠️ Getting Started

### 1. Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed, then sync the environment:

```bash
uv sync
```

### 2. Generate Model Artifacts

Before running the API, you must train the model and save the preprocessing statistics:

```bash
# 1. Clean data and save the preprocessor statistics
uv run data_cleaning.py

# 2. Train models and save the best one
uv run modeling.py
```

### 3. Run the API

Start the development server:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🐳 Docker Usage

You can also run the API as a containerized service.

### 1. Build the Image
```bash
docker build -t weather-api .
```

### 2. Run the Container
Since the model artifacts are generated during development, mount the `assets/` directory to share the trained model with the container:

```bash
docker run -p 8000:8000 -v $(pwd)/assets:/app/assets weather-api
```

## 🔌 API Usage

### Health Check
`GET http://localhost:8000/api/v1/health`



