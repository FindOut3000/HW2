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

## Docker 로 실행하기

로컬에 Python 을 설치하지 않고 바로 실행할 수 있습니다.
모델 가중치는 이미지 안에 포함되어 있어 **실행 시 모델 다운로드가 필요 없습니다.**

```bash
docker run --rm -p 8000:8000 <DOCKERHUB_USERNAME>/cat-breed-estimator:latest
```

직접 빌드하려면:

```bash
docker build -t cat-breed-estimator .
```

```bash
docker run --rm -p 8000:8000 cat-breed-estimator
```

| 항목 | 값 |
|---|---|
| 베이스 이미지 | `python:3.12.8-slim-bookworm` |
| 이미지 크기 | 약 1.84GB (모델 341MB 포함) |
| 실행 사용자 | `appuser` (비root, uid 10001) |
| 노출 포트 | `8000` |
| 헬스체크 | `/api/health` 가 `ready: true` 를 반환하는지 30초 간격 확인 |

환경 변수 `HOST`(기본 `0.0.0.0`), `PORT`(기본 `8000`) 로 바인딩을 바꿀 수 있습니다.

### 의존성 고정

리눅스에서 PyPI 의 기본 `torch` 휠은 CUDA 런타임을 포함해 이미지가 수 GB 로 커집니다.
그래서 Docker 빌드는 `requirements-docker.txt` 의 CPU 전용 휠을 사용합니다.

| | 로컬 (macOS) | Docker (Linux) |
|---|---|---|
| 의존성 파일 | `requirements.txt` | `requirements-docker.txt` |
| torch | `2.15.0` | `2.15.0+cpu` |
| torchvision | `0.29.2` | `0.29.1+cpu` |

`torch==2.15.0+cpu` 같은 로컬 버전 태그는 PyPI 에 존재하지 않으므로 PyTorch CPU 인덱스에서만
해석되며, 결과적으로 해석 결과가 결정론적으로 고정됩니다.
빌드 중 설치된 버전이 위 고정값과 일치하는지 자동으로 검사하고, 다르면 빌드가 실패합니다.

## CI / 자동 배포

`.github/workflows/ci.yml` 이 `main` 브랜치 push 시 다음을 수행합니다.

1. `linux/amd64` 이미지 빌드 — 이 과정에서 **의존성 버전 검증**과 **모델 로드 검증**(가중치를 올리고 순전파 1회 실행)이 수행됩니다.
2. 컨테이너를 띄워 스모크 테스트 — 헬스체크, 비root 실행, 라벨 48개, 추론 응답, 오류 처리(415/400), UI 200 확인
3. 검증을 모두 통과한 경우에만 Docker Hub 로 업로드

Pull Request 에서는 빌드와 검증만 수행하고 업로드하지 않습니다.

CI 가 업로드까지 성공하면 `.github/workflows/deploy.yml` 이 `workflow_run` 으로 이어서 실행되어,
집에 있는 macOS self-hosted runner 가 로컬 컨테이너를 새 이미지로 교체합니다.

### 무중단 배포 (blue-green)

8000 번은 nginx 리버스 프록시(`cat-breed-proxy`)가 계속 잡고 있고, 그 뒤에서
`cat-breed-estimator-blue` 와 `-green` 을 번갈아 씁니다.

1. 새 색 컨테이너를 **호스트 포트 없이** 띄우고 `docker exec` 로 내부 헬스를 확인합니다.
   이 동안 기존 색이 계속 트래픽을 받습니다.
2. nginx upstream 을 새 색으로 바꾸고 `nginx -t` 로 검증한 뒤 `nginx -s reload` 합니다.
   기존 워커가 처리 중인 요청을 끝내고 교체되므로 연결이 끊기지 않습니다.
3. 5초 드레인 후 이전 색을 제거합니다.
4. 검증에 실패하면 이전 `nginx.conf` 로 되돌려 reload 하고 새 색만 제거합니다.
   이전 색이 계속 살아 있으므로 롤백에도 공백이 없습니다.

접속 주소는 `http://localhost:8000` 으로 바뀌지 않습니다.
프록시가 아직 없는 최초 1회 전환에서만 8000 을 넘겨받느라 짧은 공백이 생깁니다.

배포 워크플로는 **서비스를 건드리기 전에** 독립된 모니터링 프로세스를 먼저 띄웁니다.
고정 테스트 이미지(`scripts/probe_image.py` 가 생성, 항상 같은 바이트)를 실제 `/api/analyze` 에
보내고, 응답이 끝나면 1초를 쉬었다가 다음 요청을 보냅니다. 배포 전 정상 응답을 확인한 뒤에야
교체를 시작하고, 교체 중과 완료 후 30초까지 관측을 유지합니다.

요청마다 시작·종료 시각, 응답 시간, HTTP 상태, 분석 정상 여부, **그 응답을 돌려준 배포 버전**
(`version.build`)을 기록합니다. 결과는 Actions 로그에 요청 단위로 출력되고, 실행 Summary 에
검증 주소·시간, 구간별 대표 응답, 버전 전환, 성공/실패 건수, 실제 요청 간격과 최대 기록 공백이
정리됩니다. 전체 로그는 UTF-8 로 Artifacts(`deploy-verification-<번호>`) 에 올라가며,
배포가 실패해도 로그와 요약은 남습니다.

### 필요한 GitHub Secrets

저장소의 **Settings → Secrets and variables → Actions** 에 다음을 등록해야 합니다.

| 이름 | 설명 |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub 사용자명 (이미지 이름 `<사용자명>/cat-breed-estimator` 에도 사용) |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token (비밀번호가 아닌 토큰 권장) |

시크릿이 없으면 업로드 단계는 건너뛰고 빌드·검증만 수행합니다.

### 생성되는 태그

| 조건 | 태그 |
|---|---|
| `main` 브랜치 push | `latest`, `main`, `sha-<커밋>` |
| `v*` 태그 push | `<버전>`, `latest` |

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
├── requirements.txt    의존성 목록 (로컬 / macOS)
├── requirements-docker.txt  의존성 목록 (Docker / Linux CPU)
├── Dockerfile          CPU 전용 이미지 정의
├── .dockerignore
├── .github/workflows/
│   ├── ci.yml          빌드 · 검증 · Docker Hub 업로드
│   └── deploy.yml      self-hosted runner 로컬 배포 + 무중단 검증
├── scripts/
│   ├── probe_image.py    배포 검증용 고정 테스트 이미지 생성
│   ├── deploy_monitor.py 배포 중 가용성 모니터 (독립 프로세스)
│   └── deploy_summary.py 모니터 로그 → Actions 로그 / Step Summary
├── eval/
│   ├── fetch_data.py   공개 데이터 100장 수집 + 30/70 분할
│   ├── run_eval.py     평가 실행
│   ├── compare.py      변경 전/후 비교표 생성
│   ├── RESULTS.md      성능 비교 결과
│   ├── out/            평가 결과 JSON
│   └── data/           (git 제외) 평가 이미지 — fetch_data.py 로 재현
├── static/
│   ├── index.html      UI 구조
│   ├── style.css       스타일
│   └── app.js          업로드·미리보기·결과 렌더링
├── models/             (git 제외) prepare_models.py 가 생성
└── vendor/             (git 제외) A안 설치 시 생성
```

## 모델 업데이트 (v1.1.0) — 전처리 개선

재학습 없이 **추론 전처리만** 바꿔 정확도를 올렸습니다. 가중치는 그대로입니다.

오류 분석에서 두 가지 원인을 찾았습니다.

| 문제 | 원인 |
|---|---|
| 품종 오답 | 품종 전처리가 `size:{224,224}` 고정이라 **종횡비를 무시하고 늘림**. 오답 이미지의 평균 종횡비 1.45 (정답 1.15) |
| "고양이를 찾지 못함" 오판 | 게이트 전처리가 짧은 변 256 → **중앙 224 크롭** 이라, 긴 사진에서 고양이가 프레임 밖으로 잘림 |

적용한 개선(`PREPROCESS_MODE=v2`):

- 게이트: 기존 중앙 크롭 뷰 + 전체가 보이는 정사각형 패딩 뷰, 두 뷰 점수의 **max**
  (탐지 단계라 평균은 오히려 재현율을 떨어뜨립니다 — 시도 1회차에서 확인)
- 품종: 기존에 늘린 뷰(학습 분포와 동일) + 종횡비를 보존한 패딩 뷰, 두 뷰 확률의 **평균**

평가 70장(튜닝에 사용하지 않은 분할) 기준 변경 전 → 후:

| 지표 | 변경 전 | 변경 후 |
|---|---:|---:|
| top-1 정확도 | 81.4% | **84.3%** |
| 고양이 인식 성공률 | 88.6% | **91.4%** |
| 야생 고양이과 오판율 | 2.9% | **0.0%** |
| 평균 추론 지연 | 92ms | 205ms |

자세한 내용은 [eval/RESULTS.md](eval/RESULTS.md) 를 보세요.
평가 데이터는 `timm/oxford-iiit-pet` (CC BY-SA 4.0, 공개) 에서 고양이 12품종 100장을 받아
튜닝 30장 / 평가 70장으로 나눴고, 이미지 파일은 커밋하지 않습니다
(`python3 eval/fetch_data.py` 로 재현).

`PREPROCESS_MODE` 환경변수로 `legacy` 를 주면 1.0.0 과 동일하게 동작합니다.

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
