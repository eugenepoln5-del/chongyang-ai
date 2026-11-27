# 第三方组件

崇阳AI口型自己的代码使用 MIT 许可证，见 `LICENSE`。

运行时还会用到这些组件。它们各自的许可证没有被本仓库改写：

| 组件 | 用途 | 许可证 |
| --- | --- | --- |
| MediaPipe Face Landmarker 模型 `models/face_landmarker.task` | 每帧定位人脸 | Apache-2.0（Google） |
| mediapipe | 调用上述模型 | Apache-2.0 |
| opencv-contrib-python | 摄像头、图像变形 | Apache-2.0 |
| numpy | 数值计算 | BSD-3-Clause |
| PySide6 / Qt | 窗口 | LGPL。本软件通过 pip 安装的动态库使用 Qt，没有把 Qt 静态链进专有程序 |
| soundfile | 读取 wav、flac | BSD-3-Clause |
| sounddevice / PortAudio | 播放音频 | MIT / MIT |
| miniaudio | 读取 mp3 等 | MIT |
| imageio-ffmpeg 及随附的 ffmpeg 程序 | 仅在录制时编码视频 | 项目本身为 BSD；随附的 ffmpeg 可执行文件按其自身的 GPL/LGPL 作为独立程序调用 |
| pytest | 仅开发测试 | MIT |

模型文件的 SHA-256：

`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

来源：<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
