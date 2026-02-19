"""
Модульные тесты приложения: константы калибровки и конфигурация.
"""

import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module


class TestCalibrationConfig(unittest.TestCase):
    """Сценарии: конфигурация калибровки в приложении."""

    def test_calibration_map_shape(self):
        """Карта калибровки — двумерный массив (N точек, 2 координаты)."""
        m = app_module.CALIBRATION_MAP
        self.assertEqual(m.ndim, 2)
        self.assertEqual(m.shape[1], 2)
        self.assertGreater(m.shape[0], 0)

    def test_calibration_map_values_in_unit_interval(self):
        """Все точки калибровки в диапазоне [0, 1]."""
        m = app_module.CALIBRATION_MAP
        self.assertTrue(np.all(m >= 0.0), "координаты >= 0")
        self.assertTrue(np.all(m <= 1.0), "координаты <= 1")

    def test_frames_per_point_positive(self):
        """Число кадров на точку калибровки — положительное."""
        self.assertIsInstance(app_module.FRAMES_PER_POINT, int)
        self.assertGreater(app_module.FRAMES_PER_POINT, 0)


class TestAppImports(unittest.TestCase):
    """Сценарии: корректность импортов и наличие классов."""

    def test_gaze_visualization_app_class_exists(self):
        """В модуле app определён класс GazeVisualizationApp."""
        self.assertTrue(hasattr(app_module, "GazeVisualizationApp"))

    def test_main_function_exists(self):
        """В модуле app определена функция main."""
        self.assertTrue(callable(getattr(app_module, "main", None)))


if __name__ == "__main__":
    unittest.main()
