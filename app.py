"""로컬에서 동작하는 고양이 품종 추정 웹앱 (FastAPI + PyTorch/Transformers, CPU 전용).

모델은 ./models 아래에 미리 내려받은 파일만 사용하므로 실행 시 네트워크가 필요하지 않습니다.
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s"
)
log = logging.getLogger("catbreed")

ROOT = Path(__file__).resolve().parent
BREED_DIR = ROOT / "models" / "cat_breed"
GATE_DIR = ROOT / "models" / "imagenet_gate"
STATIC_DIR = ROOT / "static"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PIXELS = 50_000_000  # 디코딩 폭탄 방지
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}

# ImageNet-1k (+ background 클래스) 기준 인덱스.
# google/mobilenet_v2_1.0_224 는 라벨이 1001개이고 0번이 background 이므로
# 통상적인 ImageNet 인덱스에 +1 된 값을 사용한다.
DOMESTIC_CAT_IDS = range(282, 287)  # tabby, tiger cat, Persian, Siamese, Egyptian cat
WILD_FELINE_IDS = range(287, 295)  # cougar ~ cheetah

CAT_PRESENT_THRESHOLD = 0.10  # 이 아래면 "고양이를 찾지 못함"으로 처리
WILD_THRESHOLD = 0.25  # 야생 고양이과로 보이는 경우의 기준

# 모델 라벨 중 품종이 아니라 털색/무늬에 해당하는 항목
PATTERN_LABELS = {
    "Calico", "Dilute Calico", "Dilute Tortoiseshell", "Tortoiseshell",
    "Torbie", "Tabby", "Tiger", "Tuxedo",
    "Domestic Long Hair", "Domestic Medium Hair", "Domestic Short Hair",
}

BREED_KO = {
    "Abyssinian": "아비시니안",
    "American Bobtail": "아메리칸 밥테일",
    "American Curl": "아메리칸 컬",
    "American Shorthair": "아메리칸 숏헤어",
    "Applehead Siamese": "애플헤드 샴",
    "Balinese": "발리니즈",
    "Bengal": "벵갈",
    "Birman": "버만",
    "Bombay": "봄베이",
    "British Shorthair": "브리티시 숏헤어",
    "Burmese": "버미즈",
    "Calico": "삼색 고양이 (캘리코)",
    "Cornish Rex": "코니시 렉스",
    "Devon Rex": "데본 렉스",
    "Dilute Calico": "연한 삼색 (딜루트 캘리코)",
    "Dilute Tortoiseshell": "연한 대모 (딜루트 토터셸)",
    "Domestic Long Hair": "장모 집고양이",
    "Domestic Medium Hair": "중모 집고양이",
    "Domestic Short Hair": "단모 집고양이",
    "Egyptian Mau": "이집션 마우",
    "Exotic Shorthair": "엑조틱 숏헤어",
    "Extra-Toes Cat - Hemingway Polydactyl": "다지증 고양이 (헤밍웨이)",
    "Havana": "하바나",
    "Himalayan": "히말라얀",
    "Japanese Bobtail": "재패니즈 밥테일",
    "Maine Coon": "메인쿤",
    "Manx": "맹크스",
    "Munchkin": "먼치킨",
    "Nebelung": "네벨룽",
    "Norwegian Forest": "노르웨이 숲",
    "Oriental Short Hair": "오리엔탈 숏헤어",
    "Persian": "페르시안",
    "Ragamuffin": "라가머핀",
    "Ragdoll": "랙돌",
    "Russian Blue": "러시안 블루",
    "Scottish Fold": "스코티시 폴드",
    "Siamese": "샴",
    "Siberian": "사이베리안",
    "Snowshoe": "스노슈",
    "Sphynx": "스핑크스",
    "Tabby": "태비 (줄무늬)",
    "Tiger": "타이거 (호랑이 무늬)",
    "Tonkinese": "통키니즈",
    "Torbie": "토비",
    "Tortoiseshell": "대모 (토터셸)",
    "Turkish Angora": "터키시 앙고라",
    "Turkish Van": "터키시 반",
    "Tuxedo": "턱시도",
}


class ModelBundle:
    """두 단계 추론 파이프라인. 1) 고양이 여부 판별 2) 품종 추정."""

    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.torch: Any = None
        self.gate_proc: Any = None
        self.gate_model: Any = None
        self.breed_proc: Any = None
        self.breed_model: Any = None
        self.breed_labels: dict[int, str] = {}

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except Exception as exc:  # pragma: no cover
            self.error = (
                f"필수 패키지를 불러오지 못했습니다 ({exc}). "
                "run.sh 로 실행했는지, vendor/ 설치가 끝났는지 확인하세요."
            )
            log.error(self.error)
            return

        for d in (BREED_DIR, GATE_DIR):
            if not (d / "config.json").exists():
                self.error = f"모델 파일이 없습니다: {d}. prepare_models.py 를 먼저 실행하세요."
                log.error(self.error)
                return

        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
            self.torch = torch

            t0 = time.perf_counter()
            self.gate_proc = AutoImageProcessor.from_pretrained(GATE_DIR)
            self.gate_model = AutoModelForImageClassification.from_pretrained(GATE_DIR)
            self.gate_model.eval()

            self.breed_proc = AutoImageProcessor.from_pretrained(BREED_DIR)
            self.breed_model = AutoModelForImageClassification.from_pretrained(BREED_DIR)
            self.breed_model.eval()
            self.breed_labels = dict(self.breed_model.config.id2label)
            took = time.perf_counter() - t0
        except Exception as exc:
            self.error = f"모델 로딩 실패: {exc}"
            log.exception("모델 로딩 실패")
            return

        # 게이트 모델의 인덱스 가정이 맞는지 확인한다.
        gate_labels = self.gate_model.config.id2label
        sample = str(gate_labels.get(284, "")).lower()
        if "persian" not in sample:
            log.warning(
                "게이트 모델 라벨 배치가 예상과 다릅니다 (284=%r). 고양이 판별이 부정확할 수 있습니다.",
                gate_labels.get(284),
            )

        self.ready = True
        log.info(
            "모델 준비 완료 (%.1fs) — 품종 %d종, CPU 스레드 %d",
            took, len(self.breed_labels), self.torch.get_num_threads(),
        )

    def predict(self, image: Image.Image) -> dict[str, Any]:
        torch = self.torch

        with torch.inference_mode():
            gate_in = self.gate_proc(images=image, return_tensors="pt")
            gate_probs = self.gate_model(**gate_in).logits.softmax(-1)[0]

            breed_in = self.breed_proc(images=image, return_tensors="pt")
            breed_probs = self.breed_model(**breed_in).logits.softmax(-1)[0]

        domestic = float(gate_probs[list(DOMESTIC_CAT_IDS)].sum())
        wild = float(gate_probs[list(WILD_FELINE_IDS)].sum())

        k = min(5, breed_probs.numel())
        top_probs, top_idx = breed_probs.topk(k)
        predictions = [
            {
                "breed": self.breed_labels.get(int(i), str(int(i))),
                "breed_ko": BREED_KO.get(self.breed_labels.get(int(i), ""), ""),
                "probability": round(float(p), 4),
                "is_pattern": self.breed_labels.get(int(i), "") in PATTERN_LABELS,
            }
            for p, i in zip(top_probs, top_idx)
        ]

        gate_top = int(gate_probs.argmax())
        return {
            "predictions": predictions,
            "domestic_cat_score": round(domestic, 4),
            "wild_feline_score": round(wild, 4),
            "gate_top_label": str(self.gate_model.config.id2label.get(gate_top, gate_top)),
        }


MODELS = ModelBundle()


@asynccontextmanager
async def lifespan(_: FastAPI):
    MODELS.load()  # 실패해도 앱은 뜨고, /api/analyze 가 503 과 함께 원인을 알려준다.
    yield


app = FastAPI(title="고양이 품종 추정기", version="1.0.0", lifespan=lifespan)


def confidence_level(p: float) -> tuple[str, str]:
    if p >= 0.60:
        return "high", "비교적 높음"
    if p >= 0.35:
        return "medium", "보통"
    return "low", "낮음"


def open_image(raw: bytes) -> Image.Image:
    try:
        probe = Image.open(io.BytesIO(raw))
        fmt = (probe.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"지원하지 않는 이미지 형식입니다 ({fmt or '알 수 없음'}). "
                       f"지원 형식: {', '.join(sorted(ALLOWED_FORMATS))}",
            )
        w, h = probe.size
        if w * h > MAX_PIXELS:
            raise HTTPException(status_code=413, detail=f"이미지 해상도가 너무 큽니다 ({w}x{h}).")
        probe.verify()  # 손상 여부 확인 (verify 후에는 재사용 불가)
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail="이미지 파일로 인식할 수 없습니다. JPG/PNG/WEBP 등의 이미지를 올려주세요.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"이미지를 읽을 수 없습니다 (손상 가능): {exc}")

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # 휴대폰 사진 회전 보정
        return img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩에 실패했습니다: {exc}")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ready": MODELS.ready,
        "error": MODELS.error,
        "breed_count": len(MODELS.breed_labels),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "allowed_formats": sorted(ALLOWED_FORMATS),
        "models": {
            "breed": "dima806/cat_breed_image_detection (ViT-base, 48종)",
            "gate": "google/mobilenet_v2_1.0_224 (ImageNet-1k)",
        },
        "device": "cpu",
    }


@app.get("/api/breeds")
async def breeds() -> dict[str, Any]:
    items = sorted(MODELS.breed_labels.values())
    return {
        "count": len(items),
        "breeds": [
            {"breed": b, "breed_ko": BREED_KO.get(b, ""), "is_pattern": b in PATTERN_LABELS}
            for b in items
        ],
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(default=None)) -> JSONResponse:
    if MODELS.error and not MODELS.ready:
        raise HTTPException(status_code=503, detail=MODELS.error)
    if not MODELS.ready:
        raise HTTPException(status_code=503, detail="모델을 준비하는 중입니다. 잠시 후 다시 시도하세요.")

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="이미지 파일이 전송되지 않았습니다.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다. 다른 이미지를 선택해주세요.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"파일이 너무 큽니다. 최대 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 까지 업로드할 수 있습니다.",
        )

    image = open_image(raw)

    t0 = time.perf_counter()
    try:
        result = MODELS.predict(image)
    except Exception as exc:
        log.exception("추론 실패")
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {exc}")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    domestic = result["domestic_cat_score"]
    wild = result["wild_feline_score"]
    top = result["predictions"][0]

    if domestic < CAT_PRESENT_THRESHOLD and wild < CAT_PRESENT_THRESHOLD:
        status, message = "no_cat", (
            "이 사진에서 고양이를 찾지 못했습니다. "
            f"모델이 본 것은 '{result['gate_top_label']}' 에 가깝습니다. "
            "고양이의 얼굴과 몸이 잘 보이는 사진으로 다시 시도해주세요."
        )
    elif wild >= WILD_THRESHOLD and wild > domestic:
        status, message = "wild_cat", (
            "야생 고양이과 동물로 보입니다. 이 모델은 집고양이 품종만 학습했기 때문에 "
            "아래 결과는 신뢰할 수 없습니다."
        )
    else:
        level_key, level_ko = confidence_level(top["probability"])
        status = "ok"
        message = f"가장 가까운 품종은 '{top['breed_ko'] or top['breed']}' 입니다. (확신도 {level_ko})"

    level_key, level_ko = confidence_level(top["probability"])

    return JSONResponse(
        {
            "status": status,
            "message": message,
            "is_estimate": True,
            "disclaimer": (
                "이 결과는 AI 모델의 통계적 추정치입니다. 품종을 확정하는 근거가 될 수 없으며, "
                "정확한 품종 확인은 수의사나 혈통서를 통해 확인하세요."
            ),
            "predictions": result["predictions"],
            "confidence": {"level": level_key, "label": level_ko, "top_probability": top["probability"]},
            "cat_detection": {
                "domestic_cat_score": domestic,
                "wild_feline_score": wild,
                "gate_top_label": result["gate_top_label"],
            },
            "image": {
                "filename": file.filename,
                "content_type": file.content_type,
                "width": image.width,
                "height": image.height,
                "size_bytes": len(raw),
            },
            "model": {
                "breed": "dima806/cat_breed_image_detection",
                "breed_note": "ViT-base, 48개 라벨, 공개된 보고 정확도 약 77%",
                "gate": "google/mobilenet_v2_1.0_224",
                "device": "cpu",
            },
            "elapsed_ms": elapsed_ms,
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
