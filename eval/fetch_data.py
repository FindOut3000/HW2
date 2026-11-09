"""공개 데이터셋에서 고양이 품종 평가용 이미지 100장을 내려받아 30/70 으로 나눈다.

데이터셋: timm/oxford-iiit-pet  (Hugging Face)
  라이선스: CC BY-SA 4.0 — 출처 표시와 동일 조건 변경 허락 하에 재사용 가능
  게이트: 없음 (공개), 원출처: Oxford-IIIT Pet Dataset (Parkhi et al., 2012)
  이미지 파일 자체는 저장소에 커밋하지 않는다(.gitignore). 이 스크립트로 재현한다.

Oxford-IIIT Pet 의 고양이 12품종은 모두 앱의 48개 라벨에 1:1 로 대응한다.

실행: python3 eval/fetch_data.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMAGES = DATA / "images"

DATASET = "timm/oxford-iiit-pet"
CONFIG = "default"
SPLIT = "test"          # 학습에 쓰이지 않은 공식 test 분할만 사용한다
SERVER = "https://datasets-server.huggingface.co"

# 데이터셋 라벨(소문자/언더스코어) -> 앱 모델의 라벨(48종 중 하나)
CAT_BREEDS = {
    "abyssinian": "Abyssinian",
    "bengal": "Bengal",
    "birman": "Birman",
    "bombay": "Bombay",
    "british_shorthair": "British Shorthair",
    "egyptian_mau": "Egyptian Mau",
    "maine_coon": "Maine Coon",
    "persian": "Persian",
    "ragdoll": "Ragdoll",
    "russian_blue": "Russian Blue",
    "siamese": "Siamese",
    "sphynx": "Sphynx",
}

TOTAL = 100
TUNE_N = 30          # 개선 시도에만 사용
SEED = 20261109      # 재현 가능한 분할


def get_json(url: str, tries: int = 5) -> dict:
    """datasets-server 는 일시적으로 500 을 돌려줄 때가 있어 재시도한다."""
    last = ""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d:
                last = str(d["error"])
            else:
                return d
        except Exception as exc:  # 네트워크/HTTP 오류
            last = f"{type(exc).__name__}: {exc}"
        wait = 3 * (attempt + 1)
        print(f"    재시도 {attempt + 1}/{tries} ({last}) — {wait}초 대기")
        time.sleep(wait)
    raise SystemExit(f"요청 실패: {url}\n  마지막 오류: {last}")


def page_rows(offset: int, length: int) -> dict:
    q = urllib.parse.urlencode({
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "offset": offset, "length": length,
    })
    return get_json(f"{SERVER}/rows?{q}")


def collect(per: dict[str, int]) -> dict[str, list[dict]]:
    """셔플된 test 분할을 앞에서부터 훑어 품종별로 필요한 수만큼 모은다."""
    found: dict[str, list[dict]] = {b: [] for b in per}
    offset, total, page = 0, None, 100
    while True:
        d = page_rows(offset, page)
        total = d.get("num_rows_total", 0)
        names = None
        for f in d["features"]:
            if f["name"] == "label":
                names = f["type"]["names"]
        for r in d["rows"]:
            row = r["row"]
            breed = names[row["label"]]
            if breed in found and len(found[breed]) < per[breed]:
                found[breed].append({
                    "image_id": str(row["image_id"]),
                    "src": row["image"]["src"],
                    "dataset_label": breed,
                })
        got = sum(len(v) for v in found.values())
        print(f"  스캔 {min(offset + page, total)}/{total} 행 — 수집 {got}/{sum(per.values())}")
        if all(len(found[b]) >= per[b] for b in per):
            return found
        offset += page
        if offset >= total:
            short = {b: per[b] - len(v) for b, v in found.items() if len(v) < per[b]}
            raise SystemExit(f"데이터가 모자랍니다: {short}")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    breeds = sorted(CAT_BREEDS)
    # 100 = 12품종 * 8장 + 앞의 4품종에 1장씩 추가
    extra = TOTAL - 8 * len(breeds)
    per = {b: 8 + (1 if i < extra else 0) for i, b in enumerate(breeds)}

    print(f"{DATASET} ({SPLIT}) 에서 고양이 {TOTAL}장 수집")
    found = collect(per)

    items: list[dict] = []
    for breed in breeds:
        for rec in found[breed][: per[breed]]:
            dest = IMAGES / f"{breed}__{rec['image_id']}.jpg"
            if not dest.exists():
                urllib.request.urlretrieve(rec["src"], dest)
            items.append({
                "file": dest.name,
                "dataset_label": breed,
                "expected_breed": CAT_BREEDS[breed],
                "image_id": rec["image_id"],
            })
        print(f"  내려받음 {breed:18s} {per[breed]}장")

    # 품종별로 고르게 나눈다. 평가 70장은 튜닝에 절대 쓰지 않는다.
    rng = random.Random(SEED)
    by_breed: dict[str, list[dict]] = {}
    for it in items:
        by_breed.setdefault(it["dataset_label"], []).append(it)

    tune: list[dict] = []
    for breed in breeds:
        group = by_breed[breed][:]
        rng.shuffle(group)
        tune.extend(group[: round(len(group) * TUNE_N / TOTAL)])

    rest = [it for it in items if it not in tune]
    rng.shuffle(rest)
    while len(tune) < TUNE_N:
        tune.append(rest.pop())
    while len(tune) > TUNE_N:
        rest.append(tune.pop())

    tune_files = {it["file"] for it in tune}
    for it in items:
        it["split"] = "tune" if it["file"] in tune_files else "eval"

    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "license": "CC BY-SA 4.0",
        "license_checked": "https://huggingface.co/datasets/timm/oxford-iiit-pet "
                           "(cardData.license=cc-by-sa-4.0, gated=False, private=False)",
        "source": "Oxford-IIIT Pet Dataset (Parkhi et al., CVPR 2012)",
        "seed": SEED,
        "total": len(items),
        "tune": sum(1 for i in items if i["split"] == "tune"),
        "eval": sum(1 for i in items if i["split"] == "eval"),
        "per_breed": {b: per[b] for b in breeds},
        "items": sorted(items, key=lambda i: i["file"]),
    }
    (DATA / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n총 {manifest['total']}장 — 튜닝 {manifest['tune']} / 평가 {manifest['eval']}")
    print(f"manifest: {DATA / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
