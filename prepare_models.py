"""모델 가중치를 ./models 아래로 내려받는다 (최초 1회, 약 341MB).

실행: python3 prepare_models.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "vendor").is_dir():
    sys.path.insert(0, str(ROOT / "vendor"))

REPOS = [
    ("dima806/cat_breed_image_detection", ROOT / "models" / "cat_breed"),
    ("google/mobilenet_v2_1.0_224", ROOT / "models" / "imagenet_gate"),
]
PATTERNS = ["config.json", "model.safetensors", "preprocessor_config.json"]

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit(
        "huggingface_hub 이 없습니다. 먼저 의존성을 설치하세요:\n"
        "  python3 -m pip install --target ./vendor -r requirements.txt   (격리 설치)\n"
        "  python3 -m pip install -r requirements.txt                     (일반 설치)"
    )

for repo, dest in REPOS:
    if (dest / "config.json").exists():
        print(f"건너뜀 (이미 있음): {dest}")
        continue
    print(f"내려받는 중: {repo} -> {dest}")
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=PATTERNS)

print("모델 준비 완료. 이제 ./run.sh 로 실행하세요.")
