#!/bin/bash
# 고양이 품종 추정기 실행 스크립트
#
# vendor/ 디렉터리가 있으면 그곳의 패키지를 우선 사용하고(가상환경 없이 격리 실행),
# 없으면 현재 python3 환경에 설치된 패키지를 사용합니다.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 를 찾을 수 없습니다. Python 3.10 이상을 설치하세요." >&2
  exit 1
fi

if [ -d vendor ]; then
  # vendor/ 를 최우선 경로로 두어 전역 site-packages 를 건드리지 않는다.
  export PYTHONPATH="$(pwd)/vendor${PYTHONPATH:+:$PYTHONPATH}"
  echo "→ vendor/ 의 패키지를 사용합니다."
fi

if ! python3 -c "import fastapi, torch, transformers" >/dev/null 2>&1; then
  cat >&2 <<'MSG'
필요한 패키지가 설치되어 있지 않습니다. 둘 중 하나를 실행하세요.

  # A. 기존 파이썬 환경을 건드리지 않는 격리 설치 (권장)
  python3 -m pip install --target ./vendor -r requirements.txt

  # B. 현재 환경에 일반 설치
  python3 -m pip install -r requirements.txt
MSG
  exit 1
fi

if [ ! -f models/cat_breed/config.json ] || [ ! -f models/imagenet_gate/config.json ]; then
  echo "모델 파일이 없습니다. 먼저 'python3 prepare_models.py' 를 실행하세요. (약 341MB 다운로드)" >&2
  exit 1
fi

# 추론은 전부 로컬 파일로 수행 — 실행 중 네트워크 호출을 막는다.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PORT="${PORT:-8000}"

echo "→ http://127.0.0.1:${PORT} 에서 실행합니다. 종료는 Ctrl+C"
exec python3 app.py
