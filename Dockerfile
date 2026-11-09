# syntax=docker/dockerfile:1.7
#
# 고양이 품종 추정 웹앱 — CPU 전용 이미지
#
# 빌드:  docker build -t cat-breed-estimator .
# 실행:  docker run --rm -p 8000:8000 cat-breed-estimator
#
# 모델 가중치(약 341MB)는 빌드 시점에 이미지 안으로 포함되므로
# 컨테이너 실행 시에는 네트워크가 필요하지 않다.

# ──────────────────────────────────────────────────────────────
# 1단계: 의존성 설치 + 모델 내려받기
# ──────────────────────────────────────────────────────────────
FROM python:3.12.8-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# 의존성만 먼저 복사해 레이어 캐시를 살린다.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 설치된 버전이 의도한 고정 버전과 일치하는지 빌드 중에 확인한다.
RUN python - <<'PY'
import importlib.metadata as md
expected = {
    "fastapi": "0.162.0", "uvicorn": "0.54.0", "python-multipart": "0.0.33",
    "pillow": "13.0.0", "torch": "2.15.0+cpu", "torchvision": "0.29.1+cpu",
    "transformers": "5.22.1", "huggingface-hub": "2.4.0",
}
bad = []
for pkg, want in expected.items():
    got = md.version(pkg)
    print(f"  {pkg:18s} {got}")
    if got != want:
        bad.append(f"{pkg}: 기대 {want}, 실제 {got}")
if bad:
    raise SystemExit("버전 불일치:\n  " + "\n  ".join(bad))
import torch
if torch.cuda.is_available() or "+cpu" not in torch.__version__:
    raise SystemExit(f"CPU 전용 빌드가 아님: {torch.__version__}")
print("의존성 버전 검증 통과 (CPU 전용)")
PY

# 런타임에 쓰지 않는 파일 제거 — torch 테스트(116MB)/헤더(64MB), 바이트코드 캐시(228MB)
RUN SP=/usr/local/lib/python3.12/site-packages && \
    rm -rf "$SP/torch/test" "$SP/torch/include" "$SP/torchvision/test" && \
    find "$SP" -name '__pycache__' -type d -prune -exec rm -rf {} + && \
    find "$SP" -name '*.pyc' -delete && \
    du -sh "$SP"

# 모델 가중치 내려받기 (약 341MB)
COPY prepare_models.py .
RUN python prepare_models.py && \
    test -f models/cat_breed/config.json && \
    test -f models/imagenet_gate/config.json && \
    du -sh models/*

# ──────────────────────────────────────────────────────────────
# 2단계: 런타임
# ──────────────────────────────────────────────────────────────
FROM python:3.12.8-slim-bookworm AS runtime

# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: 이미지 안의 모델 파일만 사용 (실행 중 네트워크 차단)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/hf \
    HOST=0.0.0.0 \
    PORT=8000

# curl 은 HEALTHCHECK 에서 사용한다.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY --from=builder /build/models ./models
COPY app.py ./
COPY static ./static

# ──────────────────────────────────────────────────────────────
# 빌드 시점 모델 로드 검증
#   단순 import 가 아니라 실제로 가중치를 올리고 추론 1회를 수행해,
#   깨진 이미지가 레지스트리로 올라가는 것을 막는다.
# ──────────────────────────────────────────────────────────────
RUN python - <<'PY'
import sys
sys.path.insert(0, "/app")
from PIL import Image
import app

app.MODELS.load()
if not app.MODELS.ready:
    raise SystemExit(f"모델 로드 실패: {app.MODELS.error}")

# 합성 이미지로 순전파 1회 — 전처리~분류까지 경로 전체를 확인한다.
probe = Image.new("RGB", (320, 320), (127, 127, 127))
result = app.MODELS.predict(probe)
top = result["predictions"][0]
assert len(result["predictions"]) == 5, result["predictions"]
assert 0.0 <= top["probability"] <= 1.0, top
assert len(app.MODELS.breed_labels) == 48, len(app.MODELS.breed_labels)
print(f"모델 로드 검증 통과 — 라벨 {len(app.MODELS.breed_labels)}개, "
      f"추론 1회 성공 (top={top['breed']} {top['probability']:.3f})")
PY

# 루트가 아닌 사용자로 실행.
# 앱은 디스크에 쓰지 않으므로 소유권 변경이 필요 없다.
# (chown -R 은 /app 342MB 를 통째로 복제하는 레이어를 만든다)
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" | grep -q '"ready": *true' || exit 1

CMD ["python", "app.py"]
