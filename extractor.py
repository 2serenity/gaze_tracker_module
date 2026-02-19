"""Извлечение признаков взгляда из кадра (MediaPipe Face Mesh)."""

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE = np.array(list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE))[:, 0]
RIGHT_EYE = np.array(list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE))[:, 0]


class GazeExtractor:
    """Извлечение вектора признаков (ландмарки глаз + масштаб/смещение головы)."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._ref = None  # (head_x, head_y, face_w, face_h)

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Возвращает вектор признаков (key_points) или None, если лицо не найдено.
        Формат: конкатенация left_eye, right_eye, [scale_x, scale_y], head_offset -> (N, 2).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 1)
        h, w = rgb.shape[:2]
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0]
        points = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float64)
        left = points[LEFT_EYE]
        right = points[RIGHT_EYE]
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        face_w = max_x - min_x
        face_h = max_y - min_y
        head_x, head_y = min_x, min_y
        if self._ref is None:
            self._ref = (head_x, head_y, face_w, face_h)
        ref_x, ref_y, ref_w, ref_h = self._ref
        scale_x = ref_w / (face_w + 1e-9)
        scale_y = ref_h / (face_h + 1e-9)
        head_offset = np.array([[head_x - ref_x, head_y - ref_y]])
        center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        left_c = left - center
        right_c = right - center
        left_c = left_c * scale_x
        right_c = right_c * scale_x
        scale_row = np.array([[scale_x, scale_y]])
        key_points = np.concatenate([left_c, right_c, scale_row, head_offset], axis=0).astype(np.float64)
        return key_points

    def reset_reference(self):
        """Сброс опорной позиции головы (для новой калибровки)."""
        self._ref = None
