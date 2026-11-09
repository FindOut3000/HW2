# 모델 업데이트 성능 비교

재학습 없이 **추론 전처리만** 바꾼 개선의 변경 전/후 비교입니다.

## 평가 데이터

- 데이터셋: `timm/oxford-iiit-pet` (test 분할) — Oxford-IIIT Pet Dataset (Parkhi et al., CVPR 2012)
- 라이선스: **CC BY-SA 4.0** · 게이트 없음 · 공개
  - 확인: https://huggingface.co/datasets/timm/oxford-iiit-pet (cardData.license=cc-by-sa-4.0, gated=False, private=False)
- 총 100장 (고양이 12품종, 품종당 8~9장) — 튜닝 30장 / 최종 평가 70장, seed `20261109`
- 아래 수치는 **평가 70장**에서만 산출했고, 이 분할은 개선 과정에 한 번도 사용하지 않았습니다.
- 이미지 파일은 커밋하지 않습니다. `python3 eval/fetch_data.py` 로 동일하게 재현됩니다.

## 결과 (평가 70장, 동일 데이터·동일 지표)

| 지표 | 변경 전 (legacy) | 변경 후 (v2) | 차이 |
| --- | ---: | ---: | ---: |
| top-1 정확도 | 81.4% | 84.3% | +2.9p ▲ |
| top-5 정확도 | 98.6% | 98.6% | +0.0p |
| 고양이 인식 성공률 (status=ok) | 88.6% | 91.4% | +2.9p ▲ |
| '고양이 없음' 오판율 | 8.6% | 8.6% | +0.0p |
| '야생 고양이과' 오판율 | 2.9% | 0.0% | -2.9p ▲ |
| 무늬 라벨이 1위인 비율 | 1.4% | 1.4% | +0.0p |
| 평균 추론 지연 | 92ms | 205ms | +113ms |

- top-1: **57/70 → 59/70**
- 새로 맞힌 이미지 3건, 새로 틀린 이미지 1건
  - 교정: `maine_coon__Maine_Coon_263.jpg` — American Curl → **Maine Coon** (정답 Maine Coon)
  - 교정: `maine_coon__Maine_Coon_83.jpg` — Norwegian Forest → **Maine Coon** (정답 Maine Coon)
  - 교정: `ragdoll__Ragdoll_45.jpg` — Himalayan → **Ragdoll** (정답 Ragdoll)
  - 퇴행: `ragdoll__Ragdoll_85.jpg` — Ragdoll → Siamese (정답 Ragdoll)

추론 지연이 2배로 늘어난 것은 두 모델 모두 뷰를 2개씩 보기 때문입니다. CPU 단일 요청 기준 여전히 1초 미만이고, 서비스 응답 시간에 체감될 수준은 아닙니다.

## 품종별 top-1

| 품종 | n | 변경 전 | 변경 후 |
| --- | ---: | ---: | ---: |
| Abyssinian | 6 | 100% | 100% |
| Bengal | 6 | 100% | 100% |
| Birman | 6 | 83% | 83% |
| Bombay | 6 | 83% | 83% |
| British Shorthair | 5 | 60% | 60% |
| Egyptian Mau | 5 | 100% | 100% |
| Maine Coon | 6 | 33% | 67% ▲ |
| Persian | 6 | 67% | 67% |
| Ragdoll | 6 | 67% | 67% |
| Russian Blue | 6 | 100% | 100% |
| Siamese | 6 | 83% | 83% |
| Sphynx | 6 | 100% | 100% |

## 개선 과정 (시도 2회, 튜닝 30장에서만 판단)

| 시도 | 내용 | 튜닝 top-1 | 튜닝 '고양이 없음' | 채택 |
| --- | --- | ---: | ---: | --- |
| — | 변경 전 (legacy) | 90.0% | 10.0% | 기준 |
| 1 | 게이트 2뷰 **평균** + 품종 좌우반전 TTA | 86.7% | 13.3% | ✗ 악화 |
| 2 | 게이트 2뷰 **max** + 품종 2뷰 평균(늘림+패딩) | 93.3% | 10.0% | ✓ 채택 |

