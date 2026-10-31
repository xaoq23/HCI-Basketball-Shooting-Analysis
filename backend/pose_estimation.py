"""MediaPipe PoseLandmarker wrapper returning named 2D landmarks."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from utils import LANDMARK_NAMES, decode_image_bytes

# Apache-2.0 model from https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"


class PoseNotFoundError(ValueError):
    """Raised when no human pose is detected in a frame."""


@dataclass(frozen=True)
class PoseFrame:
    landmarks: dict[str, dict[str, float]]
    width: int
    height: int


class PoseEstimator:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"pose model not found: {model_path}; download pose_landmarker_full.task into backend/models/"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, image_bytes: bytes) -> PoseFrame:
        image = decode_image_bytes(image_bytes)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            raise PoseNotFoundError("no pose detected; upload a clear full-body frame")
        height, width = image.shape[:2]
        landmarks = {
            name: {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
                "presence": float(lm.presence),
            }
            for name, lm in zip(LANDMARK_NAMES, result.pose_landmarks[0])
        }
        return PoseFrame(landmarks=landmarks, width=width, height=height)


@lru_cache(maxsize=1)
def get_estimator() -> PoseEstimator:
    return PoseEstimator()
