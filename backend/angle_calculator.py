"""Shooting metrics from 2D pose landmarks, computed in pixel space."""

from __future__ import annotations

import math
from typing import Literal

from pose_estimation import PoseFrame
from utils import Point, angle_degrees

ShootingHand = Literal["left", "right"]

MIN_VISIBILITY = 0.5

_SIDE_LANDMARKS = {
    "left": {
        "shoulder": "left_shoulder",
        "elbow": "left_elbow",
        "wrist": "left_wrist",
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
    },
    "right": {
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
    },
}


class InsufficientLandmarksError(ValueError):
    """Raised when the selected arm/leg landmarks are not visible enough."""


def detect_shooting_hand(landmarks: dict[str, dict[str, float]]) -> ShootingHand:
    """Pick the hand whose wrist is higher in the release frame."""
    left = landmarks["left_wrist"]
    right = landmarks["right_wrist"]
    if left["visibility"] < MIN_VISIBILITY and right["visibility"] >= MIN_VISIBILITY:
        return "right"
    if right["visibility"] < MIN_VISIBILITY and left["visibility"] >= MIN_VISIBILITY:
        return "left"
    return "left" if left["y"] < right["y"] else "right"


def _pixel(landmarks: dict[str, dict[str, float]], name: str, width: int, height: int) -> Point:
    lm = landmarks[name]
    return lm["x"] * width, lm["y"] * height


def _require_visible(landmarks: dict[str, dict[str, float]], names: list[str]) -> None:
    hidden = [name for name in names if landmarks[name]["visibility"] < MIN_VISIBILITY]
    if hidden:
        raise InsufficientLandmarksError(
            f"landmarks not visible: {', '.join(hidden)}; choose the other shooting_hand or a clearer side-view frame"
        )


def compute_frame_angles(
    frame: PoseFrame,
    shooting_hand: ShootingHand,
    *,
    include_release_angle: bool = False,
) -> dict[str, float]:
    names = _SIDE_LANDMARKS[shooting_hand]
    _require_visible(frame.landmarks, list(names.values()))

    shoulder = _pixel(frame.landmarks, names["shoulder"], frame.width, frame.height)
    elbow = _pixel(frame.landmarks, names["elbow"], frame.width, frame.height)
    wrist = _pixel(frame.landmarks, names["wrist"], frame.width, frame.height)
    hip = _pixel(frame.landmarks, names["hip"], frame.width, frame.height)
    knee = _pixel(frame.landmarks, names["knee"], frame.width, frame.height)
    ankle = _pixel(frame.landmarks, names["ankle"], frame.width, frame.height)

    angles: dict[str, float] = {}
    if include_release_angle:
        # Forearm elevation above horizontal; pixel coords keep non-square images from skewing it.
        angles["release_angle"] = round(
            math.degrees(math.atan2(abs(wrist[1] - elbow[1]), abs(wrist[0] - elbow[0]))), 1
        )
    angles["elbow_angle"] = round(angle_degrees(shoulder, elbow, wrist), 1)
    angles["knee_angle"] = round(angle_degrees(hip, knee, ankle), 1)
    # Absolute lean of the shooting-side shoulder-hip line from vertical.
    angles["trunk_tilt"] = round(
        math.degrees(math.atan2(abs(shoulder[0] - hip[0]), abs(hip[1] - shoulder[1]))), 1
    )
    return angles
