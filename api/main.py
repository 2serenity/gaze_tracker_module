"""
REST API для определения направления взгляда на мониторе.
Использует свой экстрактор (MediaPipe) и калибратор (Ridge). Без внешних библиотек трекинга.
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.extractor import GazeExtractor
from core.calibrator import GazeCalibrator

app = FastAPI(
    title="Gaze Tracker API",
    description="Определение направления взгляда на мониторе (MediaPipe + Ridge).",
    version="1.0.0",
)

_extractor = None
_calibrator = None
_screen_size = (1920, 1080)


@app.on_event("startup")
async def startup():
    global _extractor, _calibrator
    _extractor = GazeExtractor()
    _calibrator = GazeCalibrator()
    calib_path = getattr(app.state, "calib_path", None)
    if calib_path and Path(calib_path).exists():
        _calibrator = GazeCalibrator.load(calib_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "calibration_loaded": _calibrator is not None and _calibrator.fitted,
    }


class GazeResponse(BaseModel):
    x: float
    y: float
    x_norm: float
    y_norm: float


class FeaturesResponse(BaseModel):
    features: List[float]
    dim: int


def _get_point(key_points: np.ndarray) -> tuple:
    w, h = _screen_size[0], _screen_size[1]
    if _calibrator is None or not _calibrator.fitted:
        return 0.5 * w, 0.5 * h
    x_norm, y_norm = _calibrator.predict(key_points)
    return float(x_norm) * w, float(y_norm) * h


@app.post("/predict", response_model=GazeResponse)
async def predict(image: UploadFile = File(...)):
    if _extractor is None or _calibrator is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    buf = np.frombuffer(await image.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Неверное изображение")
    key_points = _extractor.extract(img)
    if key_points is None:
        raise HTTPException(status_code=422, detail="Лицо не обнаружено")
    x_px, y_px = _get_point(key_points)
    return GazeResponse(
        x=x_px, y=y_px,
        x_norm=x_px / _screen_size[0],
        y_norm=y_px / _screen_size[1],
    )


@app.post("/predict/features", response_model=GazeResponse)
async def predict_features(features: List[float]):
    if _calibrator is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    arr = np.array(features, dtype=np.float64)
    key_points = arr.reshape(-1, 2)
    x_px, y_px = _get_point(key_points)
    return GazeResponse(
        x=x_px, y=y_px,
        x_norm=x_px / _screen_size[0],
        y_norm=y_px / _screen_size[1],
    )


@app.post("/extract_features", response_model=FeaturesResponse)
async def extract_features(image: UploadFile = File(...)):
    if _extractor is None:
        raise HTTPException(status_code=503, detail="Сервис не готов")
    buf = np.frombuffer(await image.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Неверное изображение")
    key_points = _extractor.extract(img)
    if key_points is None:
        raise HTTPException(status_code=422, detail="Лицо не обнаружено")
    flat = key_points.flatten().tolist()
    return FeaturesResponse(features=flat, dim=len(flat))


def run_api(host: str = "0.0.0.0", port: int = 8000, calib_path: Optional[str] = None):
    if calib_path:
        app.state.calib_path = calib_path
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--calib", "-c", default=None, help="Путь к файлу калибровки .pkl")
    args = p.parse_args()
    run_api(host=args.host, port=args.port, calib_path=args.calib)
