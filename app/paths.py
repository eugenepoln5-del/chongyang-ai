from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_FILENAME = "face_landmarker.task"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bundled_model_path() -> Path:
    return project_root() / "models" / MODEL_FILENAME


def data_dir() -> Path:
    override = os.environ.get("CHONGYANG_HOME")
    path = Path(override) if override else Path.home() / ".chongyang-ai-kouxing"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    folder = data_dir() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "app.log"


def recordings_dir() -> Path:
    if sys.platform == "darwin":
        movies = Path.home() / "Movies"
        if movies.is_dir():
            folder = movies / "崇阳AI口型"
            folder.mkdir(parents=True, exist_ok=True)
            return folder
    folder = Path.home() / "崇阳AI口型录像"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def settings_path() -> Path:
    return data_dir() / "settings.json"


def consent_path() -> Path:
    return data_dir() / "consent.json"
