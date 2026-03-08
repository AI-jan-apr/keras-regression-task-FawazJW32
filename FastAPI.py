import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# =========================
# Load Classification Model
# =========================
classification_model = tf.keras.models.load_model("cancer_model.keras")
classification_scaler = joblib.load("cancer_scaler.pkl")

# ======================
# Load Regression Model
# ======================
regression_model = tf.keras.models.load_model("house_price_model.keras")
regression_scaler = joblib.load("house_price_scaler.pkl")


# =========================
# Classification Input
# =========================
class CancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


# ======================
# Regression Input
# ======================
class HouseInput(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float
    month: float
    year: float


@app.get("/")
def home():
    return {
        "message": "API is running",
        "endpoints": [
            "/predict_classification",
            "/predict_regression"
        ]
    }


# =========================
# Classification Endpoint
# =========================
@app.post("/predict_classification")
def predict_classification(data: CancerInput):
    input_data = np.array([[
        data.mean_radius,
        data.mean_texture,
        data.mean_perimeter,
        data.mean_area,
        data.mean_smoothness,
        data.mean_compactness,
        data.mean_concavity,
        data.mean_concave_points,
        data.mean_symmetry,
        data.mean_fractal_dimension,
        data.radius_error,
        data.texture_error,
        data.perimeter_error,
        data.area_error,
        data.smoothness_error,
        data.compactness_error,
        data.concavity_error,
        data.concave_points_error,
        data.symmetry_error,
        data.fractal_dimension_error,
        data.worst_radius,
        data.worst_texture,
        data.worst_perimeter,
        data.worst_area,
        data.worst_smoothness,
        data.worst_compactness,
        data.worst_concavity,
        data.worst_concave_points,
        data.worst_symmetry,
        data.worst_fractal_dimension
    ]])

    input_scaled = classification_scaler.transform(input_data)
    prediction = classification_model.predict(input_scaled, verbose=0)
    result = int(prediction[0][0] > 0.5)

    return {"prediction": result}


# ======================
# Regression Endpoint
# ======================
@app.post("/predict_regression")
def predict_regression(data: HouseInput):
    input_data = np.array([[
        data.bedrooms,
        data.bathrooms,
        data.sqft_living,
        data.sqft_lot,
        data.floors,
        data.waterfront,
        data.view,
        data.condition,
        data.grade,
        data.sqft_above,
        data.sqft_basement,
        data.yr_built,
        data.yr_renovated,
        data.zipcode,
        data.lat,
        data.long,
        data.sqft_living15,
        data.sqft_lot15,
        data.month,
        data.year
    ]])

    input_scaled = regression_scaler.transform(input_data)
    prediction = regression_model.predict(input_scaled, verbose=0)

    return {"predicted_price": float(prediction[0][0])}