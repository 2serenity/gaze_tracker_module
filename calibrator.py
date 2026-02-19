"""Калибратор: Ridge-регрессия key_points -> (x, y) на экране. Сохранение/загрузка в pickle."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge


class GazeCalibrator:
    """Сбор пар (признаки, целевая точка), обучение Ridge, предсказание и сохранение в файл."""

    def __init__(self, alpha: float = 0.5):
        self.reg_x = Ridge(alpha=alpha)
        self.reg_y = Ridge(alpha=alpha)
        self.X: List[np.ndarray] = []
        self.Y_x: List[float] = []
        self.Y_y: List[float] = []
        self._fitted = False

    def add(self, key_points: np.ndarray, screen_x: float, screen_y: float) -> None:
        """Добавить пример для калибровки."""
        flat = key_points.flatten().reshape(1, -1)
        self.X.append(flat)
        self.Y_x.append(screen_x)
        self.Y_y.append(screen_y)

    def fit(self) -> None:
        """Обучить Ridge по накопленным данным."""
        if len(self.X) < 3:
            self._fitted = False
            return
        X = np.vstack(self.X)
        self.reg_x.fit(X, self.Y_x)
        self.reg_y.fit(X, self.Y_y)
        self._fitted = True

    def predict(self, key_points: np.ndarray) -> Tuple[float, float]:
        """Вернуть нормализованные (x, y) в [0, 1]. Если не обучен — (0.5, 0.5)."""
        if not self._fitted:
            return 0.5, 0.5
        flat = key_points.flatten().reshape(1, -1)
        x = float(np.clip(self.reg_x.predict(flat)[0], 0.0, 1.0))
        y = float(np.clip(self.reg_y.predict(flat)[0], 0.0, 1.0))
        return x, y

    @property
    def fitted(self) -> bool:
        return self._fitted

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"reg_x": self.reg_x, "reg_y": self.reg_y, "fitted": self._fitted},
                f,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GazeCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cal = cls()
        cal.reg_x = data["reg_x"]
        cal.reg_y = data["reg_y"]
        cal._fitted = data.get("fitted", True)
        return cal
