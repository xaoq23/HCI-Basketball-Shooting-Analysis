from __future__ import annotations

import pytest

from angle_calculator import (
    InsufficientLandmarksError,
    compute_frame_angles,
    detect_shooting_hand,
)
from pose_estimation import PoseFrame


def _lm(x: float, y: float, visibility: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility, "presence": 1.0}


def _frame(
    landmarks: dict[str, dict[str, float]], width: int = 100, height: int = 100
) -> PoseFrame:
    return PoseFrame(landmarks=landmarks, width=width, height=height)


def _right_side_body(wrist_y: float = 0.5) -> dict[str, dict[str, float]]:
    return {
        "right_shoulder": _lm(0.5, 0.3),
        "right_elbow": _lm(0.5, 0.5),
        "right_wrist": _lm(0.7, wrist_y),
        "right_hip": _lm(0.5, 0.7),
        "right_knee": _lm(0.5, 0.8),
        "right_ankle": _lm(0.5, 0.9),
        "left_wrist": _lm(0.7, 0.2),
    }


def test_right_angle_joints_and_upright_trunk() -> None:
    angles = compute_frame_angles(_frame(_right_side_body()), "right", include_release_angle=True)

    assert angles["elbow_angle"] == pytest.approx(90.0)
    assert angles["knee_angle"] == pytest.approx(180.0)
    assert angles["trunk_tilt"] == pytest.approx(0.0)
    assert angles["release_angle"] == pytest.approx(0.0)


def test_release_angle_respects_image_aspect_ratio() -> None:
    body = _right_side_body()
    body["right_wrist"] = _lm(0.6, 0.4)
    angles = compute_frame_angles(
        _frame(body, width=200, height=100), "right", include_release_angle=True
    )

    assert angles["release_angle"] == pytest.approx(26.6, abs=0.1)


def test_trunk_tilt_respects_image_aspect_ratio() -> None:
    body = _right_side_body()
    body["right_shoulder"] = _lm(0.5, 0.3)
    body["right_hip"] = _lm(0.55, 0.6)
    angles = compute_frame_angles(_frame(body, width=200, height=100), "right")

    assert angles["trunk_tilt"] == pytest.approx(18.4, abs=0.1)


def test_release_angle_only_on_request() -> None:
    angles = compute_frame_angles(_frame(_right_side_body()), "right")

    assert "release_angle" not in angles


def test_detect_shooting_hand_picks_higher_visible_wrist() -> None:
    body = _right_side_body()
    body["left_wrist"] = _lm(0.7, 0.4)
    body["right_wrist"] = _lm(0.7, 0.2)

    assert detect_shooting_hand(body) == "right"


def test_detect_shooting_hand_ignores_occluded_wrist() -> None:
    body = _right_side_body()
    body["left_wrist"] = _lm(0.7, 0.1, visibility=0.2)
    body["right_wrist"] = _lm(0.7, 0.4)

    assert detect_shooting_hand(body) == "right"


def test_hidden_landmark_raises() -> None:
    body = _right_side_body()
    body["right_wrist"]["visibility"] = 0.2

    with pytest.raises(InsufficientLandmarksError, match="right_wrist"):
        compute_frame_angles(_frame(body), "right")
