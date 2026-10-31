"""FastAPI entry point for two-frame shooting analysis."""

from __future__ import annotations

from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from angle_calculator import (
    InsufficientLandmarksError,
    compute_frame_angles,
    detect_shooting_hand,
)
from pose_estimation import PoseNotFoundError, get_estimator
from utils import InvalidImageError

app = FastAPI(title="Basketball Shooting Analysis API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe for the frontend and demos."""
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(
    load_frame: UploadFile = File(..., description="蓄力帧 (set point)"),
    release_frame: UploadFile = File(..., description="出手帧 (release)"),
    shooting_hand: Literal["auto", "left", "right"] = Form("auto"),
) -> dict:
    """Return 33 named keypoints and shooting angles for both frames."""
    estimator = get_estimator()
    try:
        load_pose = estimator.detect(load_frame.file.read())
        release_pose = estimator.detect(release_frame.file.read())
        hand = (
            detect_shooting_hand(release_pose.landmarks)
            if shooting_hand == "auto"
            else shooting_hand
        )
        load_angles = compute_frame_angles(load_pose, hand)
        release_angles = compute_frame_angles(release_pose, hand, include_release_angle=True)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (PoseNotFoundError, InsufficientLandmarksError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "shooting_hand": hand,
        "load": {"angles": load_angles, "keypoints": load_pose.landmarks},
        "release": {"angles": release_angles, "keypoints": release_pose.landmarks},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
