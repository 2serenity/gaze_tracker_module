"""
Интеграционные сценарии: связка экстрактор + калибратор (синтетические key_points).
"""

import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrator import GazeCalibrator


def _synthetic_key_points(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2)).astype(np.float64)


class TestExtractorCalibratorPipeline(unittest.TestCase):
    """Сценарии: pipeline «признаки → калибратор → предсказание»."""

    def test_calibrator_predict_after_fit_in_bounds(self):
        """Калибратор обучен на векторах одной размерности; предсказание в [0, 1]."""
        cal = GazeCalibrator()
        for i in range(5):
            kp = _synthetic_key_points(10, seed=i)
            cal.add(kp, 0.5, 0.5)
        cal.fit()
        kp_new = _synthetic_key_points(10, seed=99)
        x, y = cal.predict(kp_new)
        self.assertGreaterEqual(x, 0.0)
        self.assertLessEqual(x, 1.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(y, 1.0)

    def test_multiple_add_fit_predict_cycles(self):
        """Несколько циклов add → fit → predict не ломают состояние."""
        cal = GazeCalibrator()
        for cycle in range(2):
            for i in range(4):
                cal.add(_synthetic_key_points(10, seed=cycle * 10 + i), 0.25, 0.75)
            cal.fit()
        x, y = cal.predict(_synthetic_key_points(10, seed=0))
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)


if __name__ == "__main__":
    unittest.main()
