"""
Модульные тесты калибратора (Ridge): добавление данных, обучение, предсказание, сохранение/загрузка.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrator import GazeCalibrator


def _make_key_points(seed: int = 0, n_points: int = 10) -> np.ndarray:
    """Синтетический вектор признаков (N, 2)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_points, 2)).astype(np.float64)


class TestGazeCalibratorInit(unittest.TestCase):
    """Сценарии: инициализация калибратора."""

    def test_init_default(self):
        cal = GazeCalibrator()
        self.assertFalse(cal.fitted)
        self.assertEqual(len(cal.X), 0)

    def test_init_custom_alpha(self):
        cal = GazeCalibrator(alpha=1.0)
        self.assertEqual(cal.reg_x.alpha, 1.0)
        self.assertEqual(cal.reg_y.alpha, 1.0)


class TestGazeCalibratorAddFit(unittest.TestCase):
    """Сценарии: добавление примеров и обучение."""

    def test_add_accumulates_data(self):
        cal = GazeCalibrator()
        kp = _make_key_points(0)
        cal.add(kp, 0.5, 0.5)
        cal.add(kp, 0.3, 0.7)
        self.assertEqual(len(cal.X), 2)
        self.assertEqual(cal.Y_x, [0.5, 0.3])
        self.assertEqual(cal.Y_y, [0.5, 0.7])

    def test_fit_with_less_than_3_samples_does_not_fit(self):
        cal = GazeCalibrator()
        kp = _make_key_points(0)
        cal.add(kp, 0.5, 0.5)
        cal.add(kp, 0.5, 0.5)
        cal.fit()
        self.assertFalse(cal.fitted)

    def test_fit_with_3_samples_sets_fitted(self):
        cal = GazeCalibrator()
        for i in range(3):
            cal.add(_make_key_points(i), 0.3 + i * 0.1, 0.4 + i * 0.1)
        cal.fit()
        self.assertTrue(cal.fitted)


class TestGazeCalibratorPredict(unittest.TestCase):
    """Сценарии: предсказание координат взгляда."""

    def test_predict_when_not_fitted_returns_center(self):
        cal = GazeCalibrator()
        kp = _make_key_points(0)
        x, y = cal.predict(kp)
        self.assertEqual(x, 0.5)
        self.assertEqual(y, 0.5)

    def test_predict_after_fit_returns_in_0_1(self):
        cal = GazeCalibrator()
        for i in range(5):
            kp = _make_key_points(i)
            cal.add(kp, 0.2 + i * 0.15, 0.3 + i * 0.1)
        cal.fit()
        x, y = cal.predict(_make_key_points(42))
        self.assertGreaterEqual(x, 0.0)
        self.assertLessEqual(x, 1.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(y, 1.0)

    def test_predict_is_deterministic_for_same_input(self):
        cal = GazeCalibrator()
        for i in range(4):
            cal.add(_make_key_points(i), 0.5, 0.5)
        cal.fit()
        kp = _make_key_points(99)
        a = cal.predict(kp)
        b = cal.predict(kp)
        self.assertEqual(a, b)

    def test_predict_returns_floats(self):
        cal = GazeCalibrator()
        for i in range(3):
            cal.add(_make_key_points(i), 0.5, 0.5)
        cal.fit()
        kp = _make_key_points(0)
        x, y = cal.predict(kp)
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)


class TestGazeCalibratorSaveLoad(unittest.TestCase):
    """Сценарии: сохранение и загрузка калибратора."""

    def test_save_load_roundtrip_preserves_fitted(self):
        cal = GazeCalibrator()
        for i in range(4):
            cal.add(_make_key_points(i), 0.5, 0.5)
        cal.fit()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = GazeCalibrator.load(path)
            self.assertTrue(loaded.fitted)
            kp = _make_key_points(0)
            self.assertEqual(loaded.predict(kp), cal.predict(kp))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_loaded_calibrator_predictions_match_original(self):
        cal = GazeCalibrator()
        for i in range(5):
            cal.add(_make_key_points(i), 0.1 * i, 0.2 * i)
        cal.fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.pkl"
            cal.save(path)
            loaded = GazeCalibrator.load(path)
        for seed in [0, 1, 2]:
            kp = _make_key_points(seed)
            self.assertEqual(cal.predict(kp), loaded.predict(kp))


if __name__ == "__main__":
    unittest.main()
