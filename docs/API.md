# Документация API

Определение направления взгляда: свой экстрактор (MediaPipe) и калибратор (Ridge). Для предсказаний нужна калибровка (через GUI), файл передаётся при запуске: `--calib путь/к/calib.pkl`.

## Запуск

```bash
python -m api.main --host 0.0.0.0 --port 8000 --calib saved_models/calib.pkl
```

## Эндпоинты

- **GET /health** — `{"status": "ok", "calibration_loaded": true/false}`
- **POST /predict** — изображение (multipart) → `{x, y, x_norm, y_norm}` (пиксели и норма [0,1] для 1920×1080)
- **POST /predict/features** — JSON-массив признаков (key_points.flatten()) → то же
- **POST /extract_features** — изображение → `{features, dim}`

Ошибки: 400 (изображение), 422 (лицо не найдено), 503 (сервис не готов).

Документация: http://localhost:8000/docs
