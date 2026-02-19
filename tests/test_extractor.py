"""
Модульные тесты экстрактора признаков (MediaPipe): выход при отсутствии лица, сброс опоры.
"""

import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor import GazeExtractor


class TestGazeExtractorInit(unittest.TestCase):
    """Сценарии: инициализация экстрактора."""

    def test_init_creates_face_mesh(self):
        ext = GazeExtractor()
        self.assertIsNotNone(ext.face_mesh)
        self.assertIsNone(ext._ref)


class TestGazeExtractorNoFace(unittest.TestCase):
    """Сценарии: кадр без лица — extract возвращает None."""

    def test_extract_random_image_returns_none(self):
        ext = GazeExtractor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = ext.extract(frame)
        self.assertIsNone(result)

    def test_extract_black_image_returns_none(self):
        ext = GazeExtractor()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = ext.extract(frame)
        self.assertIsNone(result)

    def test_extract_valid_shape_returns_none_without_face(self):
        """Кадр 3D (H, W, 3) — допустимый формат; без лица ожидаем None."""
        ext = GazeExtractor()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        result = ext.extract(frame)
        self.assertIsNone(result)


class TestGazeExtractorReference(unittest.TestCase):
    """Сценарии: сброс опорной позиции головы."""

    def test_reset_reference_clears_ref(self):
        ext = GazeExtractor()
        ext._ref = (10.0, 20.0, 100.0, 120.0)
        ext.reset_reference()
        self.assertIsNone(ext._ref)


if __name__ == "__main__":
    unittest.main()
