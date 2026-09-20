#!/usr/bin/env bash
set -euo pipefail

# Release publishing replaces these placeholders. The routine client is fully
# open source and no platform-specific computeproof-core wheel is required.
WHEEL_URL="${COMPUTEPROOF_WHEEL_URL:-__COMPUTEPROOF_WHEEL_URL__}"
WHEEL_SHA256="${COMPUTEPROOF_WHEEL_SHA256:-__COMPUTEPROOF_WHEEL_SHA256__}"
ASSET_URL="${COMPUTEPROOF_ASSET_URL:-__COMPUTEPROOF_ASSET_URL__}"
INSTALL_ROOT="${COMPUTEPROOF_INSTALL_ROOT:-$HOME/.local/share/computeproof}"
RUN_ROOT="${COMPUTEPROOF_RUN_ROOT:-$HOME/computeproof-runs}"
PYTHON_BIN="${COMPUTEPROOF_PYTHON:-python3}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "ComputeProof 一键安装当前只支持 Linux x86_64。" >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "未发现 nvidia-smi；请先安装可用的 NVIDIA 驱动。" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "未发现 Python 3.10+。" >&2; exit 2; }
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 2)'
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available()' || { echo "当前 Python 没有可用 CUDA Torch；先激活训练环境，或设置 COMPUTEPROOF_PYTHON。" >&2; exit 2; }
if [[ "$WHEEL_URL" != https://* || ! "$WHEEL_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "该安装脚本还是发布模板：维护者必须先写入客户端 wheel HTTPS URL 与 SHA-256。" >&2
  exit 2
fi

mkdir -p "$INSTALL_ROOT" "$RUN_ROOT"
"$PYTHON_BIN" -m venv --system-site-packages "$INSTALL_ROOT/venv-rc5"
PY="$INSTALL_ROOT/venv-rc5/bin/python"
TMP_DIR="$(mktemp -d "$INSTALL_ROOT/download.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT
CLIENT_WHEEL="$TMP_DIR/computeproof-0.5.0rc5-py3-none-any.whl"
curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error "$WHEEL_URL" -o "$CLIENT_WHEEL"
printf '%s  %s\n' "$WHEEL_SHA256" "$CLIENT_WHEEL" | sha256sum --check --status || { echo "wheel SHA-256 不一致。" >&2; exit 2; }
"$PY" -m pip install --disable-pip-version-check "${CLIENT_WHEEL}[llm]"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "现有环境的 CUDA PyTorch 不可用；请按 pytorch.org 选择器修复后重试"'
"$PY" -m computeproof doctor || { echo "预检查未完成；请检查 doctor 输出后再开始检测。" >&2; exit 2; }

OUT="$RUN_ROOT/run-$(date -u +%Y%m%dT%H%M%SZ)"
ARGS=(guided --all-devices --out "$OUT")
if [[ "$ASSET_URL" != __* && -n "$ASSET_URL" ]]; then
  ARGS+=(--asset-release-url "$ASSET_URL")
fi
exec "$PY" -m computeproof "${ARGS[@]}"
