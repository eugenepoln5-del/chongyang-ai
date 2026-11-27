#!/bin/bash
# 崇阳AI口型 — install what is missing, then open the window.
# Double-click 启动.command in the project folder. Keep this terminal open.

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

fail() {
  echo ""
  echo "$1"
  echo ""
  read -r -p "按回车键关闭…" _
  exit 1
}

echo "崇阳AI口型"
echo "正在准备，第一次会稍久一些。请不要关掉这个窗口。"
echo ""

# Double-clicked .command files often miss ~/.local/bin, where uv's Python lives.
export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

PY=""
for cand in python3.12 python3.13 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      PY="$(command -v "$cand")"
      break
    fi
  fi
done

if [[ -z "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys; print("当前 Python 版本：", sys.version.split()[0])'
  fi
  fail "没有找到 Python 3.12 或 3.13。请先安装 Python 3.12： https://www.python.org/downloads/macos/  安装完成后，重新双击「启动.command」。"
fi

echo "使用 Python：$PY"
"$PY" --version

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$PY" -m venv "$ROOT/.venv" || fail "创建运行环境失败。请确认安装 Python 时包含了 pip。然后重新双击「启动.command」。"
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip >/dev/null || true
if ! python -m pip install -r "$ROOT/requirements.txt"; then
  echo ""
  echo "安装依赖失败。如果网络比较慢，可以在这个窗口里依次粘贴下面两行后再试："
  echo "source .venv/bin/activate"
  echo "python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
  fail "依赖没有装好，所以还不能打开窗口。"
fi

export PYTHONIOENCODING=utf-8
export LANG="${LANG:-zh_CN.UTF-8}"
cd "$ROOT"
echo ""
echo "正在打开崇阳AI口型。关掉这个窗口，软件也会退出。"
echo ""
python -m app
echo ""
echo "崇阳AI口型已退出。"
