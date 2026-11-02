# 🐱 고양이 품종 추정기 (Cat Breed Estimator)

사진을 올리면 **고양이 품종을 추정**해 주는 로컬 실행 웹앱입니다.
FastAPI 백엔드와 순수 HTML/CSS/JavaScript 프론트엔드로 만들어졌고,
추론은 전부 **CPU**에서 이뤄지므로 GPU 없이 일반 노트북에서 동작합니다.

> ⚠️ **결과는 추정치입니다.**
> 이 앱의 출력은 AI 모델이 계산한 확률일 뿐이며 품종을 확정하지 않습니다.
> 정확한 품종 확인은 수의사 진료나 혈통서를 통해 하세요.

## 주요 기능

- **이미지 업로드** — 클릭 선택 또는 드래그 앤 드롭
- **즉시 미리보기** — 브라우저 안에서 `blob:` URL 로 처리 (서버 저장 없음)
- **품종 추정** — 상위 5개 후보를 확률 막대로 표시, 한글/영문 품종명 병기
- **고양이 판별** — 고양이가 아닌 사진은 엉뚱한 품종을 단정하지 않고 "고양이 미검출"로 처리
- **확신도 표시** — 높음 / 보통 / 낮음 3단계와 추정 면책 문구 상시 노출
- **오류 처리** — 파일 미선택, 빈 파일, 비이미지, 손상 파일, 용량 초과를 각각 구분해 안내

## 동작 방식

품종 모델에 사진을 바로 넣으면, 강아지 사진에도 `American Curl 53%` 같은
그럴듯한 답을 내놓습니다. 그래서 **2단계 파이프라인**을 사용합니다.

| 단계 | 모델 | 역할 | 크기 |
|---|---|---|---|
| 1 | [`google/mobilenet_v2_1.0_224`](https://huggingface.co/google/mobilenet_v2_1.0_224) | 사진에 고양이가 있는지 먼저 판별 | 약 14MB |
| 2 | [`dima806/cat_breed_image_detection`](https://huggingface.co/dima806/cat_breed_image_detection) | 48개 라벨로 품종 추정 (ViT-base) | 약 327MB |

1단계에서 집고양이 점수가 낮으면 2단계 결과를 **신뢰할 수 없음으로 흐리게 표시**합니다.
야생 고양이과(사자·호랑이 등)로 보이는 경우에도 별도 안내가 나갑니다.

품종 모델의 공개된 보고 정확도는 **약 77%** 입니다. 틀릴 수 있다는 전제로 사용하세요.

## 요구 사항

- Python **3.10 이상** (3.12.8에서 검증)
- 디스크 여유 공간 약 **1.4GB** (패키지 약 1GB + 모델 약 341MB)
- 설치 시에만 인터넷 필요 — **실행할 때는 네트워크 없이 동작**합니다

## 설치 방법

```bash
git clone <이 저장소 주소>
cd <저장소 폴더>
```

의존성 설치는 두 가지 중 하나를 고르세요.

### A. 격리 설치 — 가상환경 없이 기존 환경 보호 (권장)

패키지를 프로젝트 안 `vendor/` 폴더에만 설치하므로
시스템이나 기존 프로젝트의 `site-packages` 를 전혀 건드리지 않습니다.

```bash
python3 -m pip install --target ./vendor -r requirements.txt
```

`run.sh` 가 `vendor/` 를 자동으로 인식해 `PYTHONPATH` 에 먼저 넣어 줍니다.

### B. 일반 설치

현재 파이썬 환경(또는 직접 만든 가상환경)에 그대로 설치합니다.

```bash
python3 -m pip install -r requirements.txt
```

## 모델 준비 방법

최초 1회만 실행하면 됩니다. 모델 가중치 약 **341MB**를 `models/` 아래로 내려받습니다.

```bash
python3 prepare_models.py
```

```
models/
├── cat_breed/       dima806/cat_breed_image_detection
└── imagenet_gate/   google/mobilenet_v2_1.0_224
```

이미 받아둔 파일이 있으면 건너뜁니다. 모델 파일은 용량이 커서 `.gitignore` 에 포함되어 있으므로,
저장소를 새로 받은 사람은 이 명령을 반드시 한 번 실행해야 합니다.

## 실행 방법

```bash
./run.sh
```

브라우저에서 **<http://127.0.0.1:8000>** 접속.

- 다른 포트 사용: `PORT=8080 ./run.sh`
- 종료: 터미널에서 `Ctrl+C`
- `run.sh` 가 실행되지 않으면: `chmod +x run.sh`

스크립트를 쓰지 않고 직접 실행하려면:

```bash
export PYTHONPATH="$(pwd)/vendor"   # A안(격리 설치)으로 설치한 경우에만
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 app.py
```

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 웹 UI |
| `GET` | `/api/health` | 모델 로딩 상태 확인 |
| `GET` | `/api/breeds` | 인식 가능한 48개 라벨 목록 |
| `POST` | `/api/analyze` | `file` 필드로 이미지 업로드 → 추정 결과 JSON |

```bash
curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/api/analyze
```

응답 예시 (일부 생략):

```json
{
  "status": "ok",
  "message": "가장 가까운 품종은 '샴' 입니다. (확신도 비교적 높음)",
  "is_estimate": true,
  "predictions": [
    { "breed": "Siamese", "breed_ko": "샴", "probability": 0.9015, "is_pattern": false }
  ],
  "confidence": { "level": "high", "label": "비교적 높음", "top_probability": 0.9015 },
  "cat_detection": { "domestic_cat_score": 0.9585, "gate_top_label": "Siamese cat, Siamese" },
  "elapsed_ms": 120
}
```

`status` 값은 `ok`(품종 추정 성공), `no_cat`(고양이 미검출), `wild_cat`(야생 고양이과 추정) 중 하나입니다.

### 오류 응답

| 상황 | HTTP 코드 |
|---|---|
| 파일 미전송 / 빈 파일 / 디코딩 실패 | `400` |
| 이미지가 아닌 파일, 미지원 형식 | `415` |
| 10MB 초과, 해상도 과대 | `413` |
| 모델이 아직 준비되지 않음 | `503` |

지원 형식: JPEG · PNG · WEBP · BMP · GIF · TIFF (최대 10MB)

## 프로젝트 구조

```
.
├── app.py              FastAPI 서버 + 2단계 추론 파이프라인
├── prepare_models.py   모델 가중치 다운로드 스크립트
├── run.sh              실행 스크립트 (PYTHONPATH·오프라인 설정 처리)
├── requirements.txt    의존성 목록
├── static/
│   ├── index.html      UI 구조
│   ├── style.css       스타일
│   └── app.js          업로드·미리보기·결과 렌더링
├── models/             (git 제외) prepare_models.py 가 생성
└── vendor/             (git 제외) A안 설치 시 생성
```

## 48개 라벨에 대한 참고

라벨 중 `Tabby`, `Calico`, `Tuxedo`, `Tortoiseshell`, `Torbie`, `Tiger`,
`Domestic Short/Medium/Long Hair` 는 품종이 아니라 **털색·무늬** 분류입니다.
UI에서는 `무늬/털색` 태그를 붙여 구분해 표시합니다.

## 개인정보 처리

- 업로드한 이미지는 **메모리에서만** 처리되며 디스크에 저장하지 않습니다.
- 미리보기는 브라우저 안에서 `blob:` URL 로 처리됩니다.
- 서버는 `127.0.0.1` 에만 바인딩되어 외부에서 접근할 수 없습니다.
- 실행 중에는 외부로 어떤 요청도 보내지 않습니다 (`HF_HUB_OFFLINE=1`).

## 라이선스 / 출처

모델 가중치는 각 Hugging Face 저장소의 라이선스를 따릅니다.
사용 전 해당 저장소의 라이선스 조건을 확인하세요.
