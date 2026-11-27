"""崇阳AI口型的窗口。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.bootstrap import ModelError, resolve_model
from app.consent import CONSENT_BODY, CONSENT_CHECK, has_consent, save_consent
from app.paths import settings_path
from app.session import Session, Snapshot
from app.version import APP_NAME, ORG_NAME, __version__

AUDIO_FILTER = "音频 (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.aiff *.aif);;所有文件 (*)"
VIDEO_FILTER = "视频 (*.mp4 *.mov *.m4v *.avi *.mkv);;所有文件 (*)"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setStyle("Fusion")
    _install_font(app)
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(_app_icon())
    if not has_consent():
        dialog = ConsentDialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return 0
        save_consent()
    try:
        model = _ensure_model_ui()
    except ModelError as exc:
        QMessageBox.critical(None, APP_NAME, str(exc))
        return 1
    window = MainWindow(model)
    window.show()
    return app.exec()


class ConsentDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("使用须知")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.notice = QLabel(CONSENT_BODY)
        self.notice.setWordWrap(True)
        self.notice.setObjectName("body")
        self.checkbox = QCheckBox(CONSENT_CHECK)
        self.checkbox.setObjectName("consent")
        self.checkbox.setMinimumHeight(48)
        self.accept_btn = QPushButton("同意并开始")
        self.accept_btn.setObjectName("primary")
        self.accept_btn.setEnabled(False)
        quit_btn = QPushButton("退出")
        self.checkbox.toggled.connect(self.accept_btn.setEnabled)
        self.accept_btn.clicked.connect(self.accept)
        quit_btn.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        title = QLabel(APP_NAME)
        title.setObjectName("brand")
        layout.addWidget(title)
        layout.addWidget(self.notice)
        layout.addWidget(self.checkbox)
        row = QHBoxLayout()
        row.addWidget(quit_btn)
        row.addWidget(self.accept_btn)
        layout.addLayout(row)


class NoticeDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("使用须知")
        self.setMinimumWidth(460)
        label = QLabel(CONSENT_BODY + "\n\n" + CONSENT_CHECK)
        label.setWordWrap(True)
        label.setObjectName("body")
        button = QPushButton("知道了")
        button.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.addWidget(label)
        layout.addWidget(button)


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"关于{APP_NAME}")
        self.setMinimumWidth(480)
        text = (
            f"{APP_NAME}  {__version__}\n\n"
            "这是一个在你自己的 Mac 上运行的摄像头口型预览。上面是原片，"
            "下面是跟着所选音频张合的合成画面。暂停或停止后，合成画面立刻回到真实嘴型。\n\n"
            "默认不上传摄像头，也不上传音频。\n\n"
            "这一版的口型是实时算出来的：用 MediaPipe 人脸网格（Apache-2.0）跟上你的脸，"
            "再按音频的响度和音色去开合、展宽或收圆嘴巴，并补上口腔和牙齿。"
            "它能跟着转头、起身和靠近，但不是逐像素重绘的神经网络。"
            "所以牙齿不会和真人完全一样，侧脸、手挡住嘴、或者嘴张得特别大时，会显得不自然。\n\n"
            "请只用于你自己的脸，或已经明确同意的人。不要用来冒充他人。\n\n"
            f"© {ORG_NAME}"
        )
        label = QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("body")
        button = QPushButton("关闭")
        button.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.addWidget(label)
        layout.addWidget(button)


class VideoPane(QFrame):
    def __init__(self, title: str, caption: str, accent: bool = False) -> None:
        super().__init__()
        self.title = title
        self.caption = caption
        self.accent = accent
        self.overlay = ""
        self._image: QImage | None = None
        self.setObjectName("accentPane" if accent else "plainPane")
        self.setMinimumHeight(220)

    def set_frame(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def set_overlay(self, text: str) -> None:
        self.overlay = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#141210"))
        target = self.rect().adjusted(10, 36, -10, -12)
        if self._image is None or self._image.isNull():
            painter.setPen(QColor("#a3988c"))
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, self.caption)
        else:
            scaled = self._image.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            x = target.x() + (target.width() - scaled.width()) // 2
            y = target.y() + (target.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e25b2a" if self.accent else "#2a2622"))
        pill = painter.fontMetrics().horizontalAdvance(self.title) + 18
        painter.drawRoundedRect(12, 10, pill, 22, 8, 8)
        painter.setPen(QColor("#fff8f2"))
        painter.drawText(12, 10, pill, 22, Qt.AlignmentFlag.AlignCenter, self.title)
        if self.overlay:
            painter.setPen(QColor("#f0c14a"))
            painter.drawText(self.rect().adjusted(12, 0, -12, -8), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, self.overlay)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1180, 820)
        self.session = Session(model_path)
        self._settings = _load_settings()
        self.session.mirror = bool(self._settings.get("mirror", True))
        self.session.strength = float(self._settings.get("strength", 1.0))
        self.session.smooth = float(self._settings.get("smooth", 0.35))
        self._slider_held = False
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        side = QWidget()
        side.setObjectName("side")
        side.setFixedWidth(300)
        form = QVBoxLayout(side)
        form.setContentsMargins(18, 18, 18, 16)
        form.setSpacing(8)

        brand = QLabel(APP_NAME)
        brand.setObjectName("brand")
        sub = QLabel("本地实时口型 · 画面不上传")
        sub.setObjectName("muted")
        form.addWidget(brand)
        form.addWidget(sub)
        form.addSpacing(8)
        form.addWidget(_section("画面"))

        self.camera_btn = QPushButton("开启摄像头")
        self.camera_btn.setObjectName("primary")
        self.camera_btn.clicked.connect(self._toggle_camera)
        self.video_btn = QPushButton("打开视频…")
        self.video_btn.clicked.connect(self._open_video)
        self.demo_btn = QPushButton("示例演示")
        self.demo_btn.clicked.connect(self._start_demo)
        self.mirror_box = QCheckBox("镜像预览")
        self.mirror_box.setChecked(self.session.mirror)
        self.mirror_box.toggled.connect(self._set_mirror)
        form.addWidget(self.camera_btn)
        form.addWidget(self.video_btn)
        form.addWidget(self.demo_btn)
        form.addWidget(self.mirror_box)

        form.addSpacing(8)
        form.addWidget(_section("音频"))
        self.audio_label = QLabel("还没有选择音频")
        self.audio_label.setObjectName("file")
        self.audio_label.setWordWrap(True)
        self.pick_btn = QPushButton("选择音频…")
        self.pick_btn.clicked.connect(self._pick_audio)
        self.example_btn = QPushButton("使用示例音频")
        self.example_btn.clicked.connect(self._example_audio)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderPressed.connect(self._hold_slider)
        self.slider.sliderReleased.connect(self._release_slider)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("muted")
        form.addWidget(self.audio_label)
        form.addWidget(self.pick_btn)
        form.addWidget(self.example_btn)
        form.addWidget(self.slider)
        form.addWidget(self.time_label)

        row = QHBoxLayout()
        self.play_btn = QPushButton("开始")
        self.play_btn.setObjectName("primary")
        self.pause_btn = QPushButton("暂停")
        self.stop_btn = QPushButton("停止")
        self.play_btn.clicked.connect(self._play)
        self.pause_btn.clicked.connect(self.session.pause)
        self.stop_btn.clicked.connect(self.session.stop_playback)
        row.addWidget(self.play_btn)
        row.addWidget(self.pause_btn)
        row.addWidget(self.stop_btn)
        form.addLayout(row)

        form.addSpacing(8)
        form.addWidget(_section("合成"))
        form.addWidget(QLabel("口型强度"))
        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(40, 160)
        self.strength.setValue(int(self.session.strength * 100))
        self.strength.valueChanged.connect(self._set_strength)
        form.addWidget(self.strength)
        form.addWidget(QLabel("平滑"))
        self.smooth = QSlider(Qt.Orientation.Horizontal)
        self.smooth.setRange(0, 100)
        self.smooth.setValue(int(self.session.smooth * 100))
        self.smooth.valueChanged.connect(self._set_smooth)
        form.addWidget(self.smooth)
        self.record_btn = QPushButton("录制合成画面")
        self.record_btn.clicked.connect(self._toggle_record)
        form.addWidget(self.record_btn)
        form.addStretch(1)

        self.status = QLabel("正在准备人脸模型…")
        self.status.setWordWrap(True)
        self.status.setObjectName("status")
        limit = QLabel("暂停或停止后，下面的画面会立刻回到真实嘴型。")
        limit.setWordWrap(True)
        limit.setObjectName("muted")
        form.addWidget(self.status)
        form.addWidget(limit)
        links = QHBoxLayout()
        notice = QPushButton("使用须知")
        about = QPushButton("关于")
        notice.setObjectName("link")
        about.setObjectName("link")
        notice.clicked.connect(lambda: NoticeDialog(self).exec())
        about.clicked.connect(lambda: AboutDialog(self).exec())
        links.addWidget(notice)
        links.addWidget(about)
        form.addLayout(links)

        stage = QWidget()
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(14, 14, 14, 14)
        stage_layout.setSpacing(10)
        self.original = VideoPane("原片", "摄像头未开启\n原片不会被改动", accent=False)
        self.synth = VideoPane("合成", "合成画面会跟着音频动嘴", accent=True)
        stage_layout.addWidget(self.original, 1)
        stage_layout.addWidget(self.synth, 1)
        outer.addWidget(side)
        outer.addWidget(stage, 1)

        self.play_btn.setShortcut("S")
        self.pause_btn.setShortcut("Space")
        self.stop_btn.setShortcut("Esc")
        self.record_btn.setShortcut("R")
        self.mirror_box.setShortcut("M")

    def _toggle_camera(self) -> None:
        if self.session._source_kind == "camera" and self.session._running:
            self.session.stop_source()
            self.camera_btn.setText("开启摄像头")
            self.original.set_frame(None)
            self.synth.set_frame(None)
            return
        message = self.session.start_camera(0)
        if message:
            QMessageBox.warning(self, APP_NAME, message)
            return
        self.camera_btn.setText("关闭摄像头")

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", str(Path.home()), VIDEO_FILTER)
        if not path:
            return
        message = self.session.open_video(Path(path))
        if message:
            QMessageBox.warning(self, APP_NAME, message)
            return
        self.camera_btn.setText("开启摄像头")

    def _start_demo(self) -> None:
        self.session.start_demo()
        self.camera_btn.setText("开启摄像头")
        if not self.session.audio.loaded:
            message = self.session.load_example_audio()
            if message:
                QMessageBox.warning(self, APP_NAME, message)

    def _pick_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择音频", str(Path.home()), AUDIO_FILTER)
        if not path:
            return
        message = self.session.load_audio(Path(path))
        if message:
            QMessageBox.warning(self, APP_NAME, message)

    def _example_audio(self) -> None:
        message = self.session.load_example_audio()
        if message:
            QMessageBox.warning(self, APP_NAME, message)

    def _play(self) -> None:
        message = self.session.play()
        if message:
            QMessageBox.information(self, APP_NAME, message)

    def _toggle_record(self) -> None:
        if self.session.recorder.active:
            _path, note = self.session.stop_recording()
            self.record_btn.setText("录制合成画面")
            self.record_btn.setObjectName("")
            self.record_btn.style().unpolish(self.record_btn)
            self.record_btn.style().polish(self.record_btn)
            if note:
                QMessageBox.information(self, APP_NAME, note)
            return
        message = self.session.start_recording()
        if message:
            QMessageBox.warning(self, APP_NAME, message)
            return
        self.record_btn.setText("停止录制")
        self.record_btn.setObjectName("recording")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)

    def _set_mirror(self, checked: bool) -> None:
        self.session.mirror = checked
        self._save_settings()

    def _set_strength(self, value: int) -> None:
        self.session.strength = value / 100.0
        self._save_settings()

    def _set_smooth(self, value: int) -> None:
        self.session.smooth = value / 100.0
        self._save_settings()

    def _hold_slider(self) -> None:
        self._slider_held = True

    def _release_slider(self) -> None:
        self._slider_held = False
        duration = self.session.audio.duration
        if duration > 0:
            self.session.seek(self.slider.value() / 1000.0 * duration)

    def _tick(self) -> None:
        snap = self.session.snapshot()
        if snap is None:
            if self.session._engine_error:
                self.status.setText(self.session._engine_error)
            elif not self.session.engine_ready:
                self.status.setText("正在准备人脸模型…")
            else:
                self.status.setText(self.session._status)
            return
        self.original.set_frame(_bgr_to_qimage(snap.original))
        self.synth.set_frame(_bgr_to_qimage(snap.synthesized))
        self.synth.set_overlay(snap.note)
        self._fill_status(snap)
        if snap.audio_name:
            self.audio_label.setText(snap.audio_name)
        if not self._slider_held and snap.duration_s > 0:
            self.slider.setValue(int(np.clip(snap.position_s / snap.duration_s, 0.0, 1.0) * 1000))
        self.time_label.setText(f"{_clock(snap.position_s)} / {_clock(snap.duration_s)}")
        self.pause_btn.setEnabled(snap.play_state == "playing")
        self.stop_btn.setEnabled(snap.play_state != "stopped")
        if snap.play_state == "paused":
            self.play_btn.setText("继续")
        else:
            self.play_btn.setText("开始")

    def _fill_status(self, snap: Snapshot) -> None:
        bits = [snap.status]
        if snap.source_label and snap.source_label != "未开启":
            bits.append(snap.source_label)
        if snap.fps > 1:
            bits.append(f"{snap.fps:.0f} 帧/秒")
        if snap.process_ms > 0:
            bits.append(f"处理 {snap.process_ms:.0f} 毫秒")
        if snap.recording:
            bits.append("录制中")
        if snap.silent_audio and snap.play_state == "playing":
            bits.append("没有声音输出")
        self.status.setText(" · ".join(bits))

    def _save_settings(self) -> None:
        payload = {
            "mirror": self.session.mirror,
            "strength": self.session.strength,
            "smooth": self.session.smooth,
        }
        settings_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        self.session.shutdown()
        super().closeEvent(event)


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return image.copy()


def _load_settings() -> dict:
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_model_ui() -> Path:
    bundled = resolve_model_if_present()
    if bundled is not None:
        return bundled
    box = QProgressDialog("正在下载人脸模型（约 4MB），只需这一次…", None, 0, 0)
    box.setWindowTitle(APP_NAME)
    box.setCancelButton(None)
    box.setWindowModality(Qt.WindowModality.ApplicationModal)
    box.show()
    error: list[Exception] = []
    holder: list[Path] = []

    def work() -> None:
        try:
            holder.append(resolve_model())
        except Exception as exc:
            error.append(exc)

    import threading

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while thread.is_alive():
        QApplication.processEvents()
        thread.join(0.05)
    box.close()
    if error:
        raise ModelError(str(error[0]))
    return holder[0]


def resolve_model_if_present() -> Path | None:
    from app.paths import MODEL_SHA256, bundled_model_path, data_dir
    from app.bootstrap import _sha256

    for candidate in (bundled_model_path(), data_dir() / "models" / "face_landmarker.task"):
        if candidate.is_file():
            try:
                if _sha256(candidate) == MODEL_SHA256:
                    return candidate
            except OSError:
                continue
    return None


def _install_font(app: QApplication) -> None:
    families = ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei"]
    for path in (
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    ):
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    font = QFont()
    font.setFamilies(families)
    font.setPointSize(13)
    app.setFont(font)


def _app_icon() -> QIcon:
    pix = QPixmap(256, 256)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#e25b2a"))
    painter.drawRoundedRect(16, 16, 224, 224, 52, 52)
    painter.setBrush(QColor("#f0c14a"))
    painter.drawEllipse(176, 36, 36, 36)
    painter.setPen(QPen(QColor("#fff8f2"), 16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawArc(68, 108, 120, 78, 200 * 16, 140 * 16)
    painter.end()
    return QIcon(pix)


STYLESHEET = """
QWidget { background: #1a1714; color: #f3efe8; font-size: 13px; }
QMainWindow { background: #12100e; }
#side { background: #1c1916; border-right: 1px solid #3a342e; }
#brand { font-size: 22px; font-weight: 600; color: #fff8f2; background: transparent; }
#muted, QLabel#muted { color: #a3988c; background: transparent; }
#section { color: #f0c14a; font-size: 12px; letter-spacing: 1px; background: transparent; margin-top: 4px; }
#file { background: #262220; border-radius: 8px; padding: 8px; }
#status { background: #262220; border-radius: 8px; padding: 8px; }
QLabel { background: transparent; }
QPushButton {
    background: #2c2824; color: #f3efe8; border: 1px solid #4a433c;
    border-radius: 8px; padding: 8px 10px; min-height: 18px;
}
QPushButton:hover { background: #3a342e; }
QPushButton:disabled { color: #6e675f; }
QPushButton#primary { background: #e25b2a; border: 1px solid #e25b2a; color: #fff8f2; font-weight: 600; }
QPushButton#primary:hover { background: #f06a38; }
QPushButton#primary:disabled { background: #3a342e; border: 1px solid #3a342e; color: #6e675f; }
QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #c4b8a8; border-radius: 4px; background: #262220; }
QCheckBox::indicator:checked { background: #e25b2a; border: 1px solid #e25b2a; }
QPushButton#recording { background: #8d2b2b; border: 1px solid #d64545; color: #fff8f2; }
QPushButton#link { background: transparent; border: none; color: #f0c14a; padding: 4px; }
QCheckBox { background: transparent; spacing: 8px; }
QSlider::groove:horizontal { height: 4px; background: #3a342e; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -6px 0; background: #f0c14a; border-radius: 7px; }
#plainPane, #accentPane { border-radius: 12px; }
#accentPane { border: 1px solid #e25b2a; }
#plainPane { border: 1px solid #3a342e; }
QDialog { background: #1c1916; }
QMessageBox { background: #1c1916; }
"""
