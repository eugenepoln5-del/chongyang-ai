# 崇阳AI口型

在 Mac 上打开摄像头，上半部分是原片，下半部分按你选的音频做实时口型。暂停或停止后，合成画面立刻回到真实的嘴。画面和音频默认只在本机处理。

给不写代码的人：请直接看 [使用说明.md](使用说明.md)。双击 `启动.command` 即可，第一次会自动安装依赖。

## 能做什么

- 开启或关闭摄像头，并排查看原片和合成画面（上下两格）
- 选择 wav、mp3、m4a 等音频，开始、暂停、停止
- 暂停和停止会马上停止口型，合成格回到摄像头里的嘴
- 镜像预览、录制合成画面
- 没有摄像头时，可以用视频，或用内置的绘制示例看口型是否在动
- 第一次打开会要求确认：只能用于自己的脸，或已经明确同意的人，不能用来冒充他人

## 这一版怎么做口型

人脸位置用 **MediaPipe Face Landmarker**（Apache-2.0，模型文件已放在 `models/face_landmarker.task`）每一帧跟踪，所以转头、靠近、起身时嘴会跟着脸走。

嘴的张开、展宽和收圆来自音频的能量和频谱，在本机实时算，不上传。口腔和牙齿是按张开程度补绘的，避免闭着的嘴被拉开后只剩一块被拉长的皮肤。

这不是神经网络逐像素重绘。好处是 Apple 芯片、没有独显也能保持交互，不需要下载数 GB 的权重，也不需要付费接口。代价是牙齿不会和真人一致，侧脸和遮挡会变差，也不会精确对应每一个音素。更重的唇形模型在很多 Mac 上做不到稳定的实时预览，所以没有放进这一版。

处理延迟大约是一帧（常见为几十毫秒）加上音频缓冲。原片本身不做嘴部修改。

## 在 Mac 上运行

需要：

- Apple 芯片，macOS 13 或更新
- Python 3.12 或 3.13（[python.org 的 macOS 安装包](https://www.python.org/downloads/macos/)）
- 第一次联网，用来安装 Python 依赖

```bash
bash 启动.command
```

或双击 `启动.command`。请保持弹出的终端开着。

依赖写在 `requirements.txt`，版本已固定。国内网络可以加清华镜像，步骤在使用说明里。

人脸模型已经包含在仓库中，校验和见 `app/paths.py`。录制若缺少 ffmpeg，会使用 imageio-ffmpeg 提供的程序；也可以自行 `brew install ffmpeg`。

这里的开发环境没有摄像头，也没有 Apple GPU，因此实机的摄像头权限需要在 Mac 上确认。在这台 Linux 测试机上，1280×720 的人脸定位大约 5–10 毫秒，嘴部变形大约 20 毫秒。Apple 芯片上通常更快。仓库里的测试覆盖口型计算、暂停时画面与原片逐像素一致、录制，以及窗口标题。

## 开发

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m app.selftest
pytest
python -m app
```

## 许可证

本仓库代码为 MIT。第三方组件和模型的许可证见 [NOTICE.md](NOTICE.md)。
