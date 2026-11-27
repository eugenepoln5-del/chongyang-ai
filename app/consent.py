from __future__ import annotations

import json
from datetime import datetime, timezone

from app.paths import consent_path

CONSENT_VERSION = 1

CONSENT_BODY = (
    "「崇阳AI口型」会根据你选的音频，在摄像头画面上实时改变嘴型。\n\n"
    "请只在下面两种情况下使用：\n"
    "1. 画面里是你自己的脸；\n"
    "2. 画面里的人已经清楚地同意你这样做。\n\n"
    "不要用它冒充别人，也不要用它做欺诈、诽谤或其他你没有权利做的事。\n\n"
    "摄像头画面和音频默认只在这台电脑上处理，不会上传。"
)

CONSENT_CHECK = "我确认只使用自己的脸，或已明确同意的人的脸，并且不会用本软件冒充他人。"


def has_consent() -> bool:
    path = consent_path()
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("accepted")) and int(data.get("version", 0)) == CONSENT_VERSION


def save_consent() -> None:
    payload = {
        "accepted": True,
        "version": CONSENT_VERSION,
        "app": "崇阳AI口型",
        "at": datetime.now(timezone.utc).isoformat(),
    }
    consent_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
