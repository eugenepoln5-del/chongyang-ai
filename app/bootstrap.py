from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from app.paths import MODEL_SHA256, MODEL_URL, bundled_model_path, data_dir


class ModelError(Exception):
    pass


def resolve_model() -> Path:
    bundled = bundled_model_path()
    if bundled.is_file() and _sha256(bundled) == MODEL_SHA256:
        return bundled
    cached = data_dir() / "models" / bundled.name
    if cached.is_file() and _sha256(cached) == MODEL_SHA256:
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    _download(MODEL_URL, cached)
    if _sha256(cached) != MODEL_SHA256:
        cached.unlink(missing_ok=True)
        raise ModelError("人脸模型下载不完整。请检查网络后重新打开崇阳AI口型。")
    return cached


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "chongyang-ai-kouxing"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ModelError(
            "下载人脸模型失败。项目里的 models 文件夹应已自带模型；"
            "如果那个文件不在，请联网后重试。"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
