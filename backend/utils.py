"""Shared helpers for image decoding and 2D geometry."""

from __future__ import annotations

import math

import cv2
import numpy as np

LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

Point = tuple[float, float]


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise InvalidImageError("uploaded file is empty")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("uploaded file is not a decodable image")
    return image


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    """Angle at b formed by the a-b-c chain, in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        raise ValueError("degenerate angle: duplicate joint coordinates")
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
