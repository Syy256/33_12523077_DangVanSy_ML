
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ==============================
# 1. Load model
# ==============================

model = joblib.load("knn_house_model.pkl")


# ==============================
# 2. Khởi tạo FastAPI
# ==============================

app = FastAPI(
    title="KNN Dự đoán giá nhà",
    description="API sử dụng KNN Regression để dự đoán giá nhà",
    version="1.0.0"
)


# ==============================
# 3. Định nghĩa dữ liệu đầu vào
# ==============================

class HousePredictionRequest(BaseModel):

    dien_tich: float = Field(..., gt=0)
    so_phong: int = Field(..., ge=1)
    khoang_cach_trung_tam: float = Field(..., ge=0)


# ==============================
# 4. API trang chủ
# ==============================

@app.get("/")
def root():

    return {
        "trang_thai": "hoat_dong",
        "ten_mo_hinh": "KNN du doan gia nha",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


# ==============================
# 5. API kiểm tra server
# ==============================

@app.get("/health")
def health():

    return {
        "trang_thai": "hoat_dong",
        "mo_hinh": "KNN Regression"
    }


# ==============================
# 6. API dự đoán giá nhà
# ==============================

@app.post("/predict")
def predict(request: HousePredictionRequest):

    input_data = pd.DataFrame([{
        "area": request.dien_tich,
        "rooms": request.so_phong,
        "distance": request.khoang_cach_trung_tam
    }])

    prediction = float(model.predict(input_data)[0])

    return {
        "gia_du_doan": round(prediction, 2),
        "don_vi": "trieu_vnd"
    }


print("Đã tạo file main.py")
