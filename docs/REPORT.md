# Отчёт. Модуль определения направления взгляда

Модуль работает **без EyeGestures и без своей нейросети**: только свои компоненты на базе открытых библиотек.

## Реализация

1. **core/extractor.py** — извлечение признаков из кадра: MediaPipe Face Mesh, ландмарки левого/правого глаза, масштаб и смещение головы (формат key_points).
2. **core/calibrator.py** — калибратор Ridge (scikit-learn): накопление пар (key_points, target_x, target_y), обучение, предсказание (норма 0..1), сохранение/загрузка в .pkl.
3. **GUI** — камера, фаза калибровки по точкам, отображение траектории взгляда, загрузка/сохранение калибровки.
4. **API** — загрузка калибровки при старте, эндпоинты /predict, /predict/features, /extract_features.

Зависимости: numpy, opencv-python, mediapipe, scikit-learn, fastapi, uvicorn, Pillow. Папка EyeGestures-main и пакет eyeGestures не используются.
