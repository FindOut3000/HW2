"""변경 전(legacy)과 변경 후(v2)의 평가 결과를 같은 데이터·같은 지표로 비교한다."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "out"


def load(name: str, split: str) -> dict:
    return json.loads((OUT / f"{name}__{split}.json").read_text(encoding="utf-8"))


def main() -> None:
    before = load("baseline", "eval")
    after = load("final_v2", "eval")
    man = json.loads((ROOT / "eval" / "data" / "manifest.json").read_text(encoding="utf-8"))

    b, a = before["summary"], after["summary"]
    br = {r["file"]: r for r in before["records"]}
    ar = {r["file"]: r for r in after["records"]}

    fixed = [f for f in br if not br[f]["correct_top1"] and ar[f]["correct_top1"]]
    broke = [f for f in br if br[f]["correct_top1"] and not ar[f]["correct_top1"]]

    L = []
    w = L.append
    w("# 모델 업데이트 성능 비교")
    w("")
    w("재학습 없이 **추론 전처리만** 바꾼 개선의 변경 전/후 비교입니다.")
    w("")
    w("## 평가 데이터")
    w("")
    w(f"- 데이터셋: `{man['dataset']}` ({man['split']} 분할) — {man['source']}")
    w(f"- 라이선스: **{man['license']}** · 게이트 없음 · 공개")
    w(f"  - 확인: {man['license_checked']}")
    w(f"- 총 {man['total']}장 (고양이 12품종, 품종당 8~9장) — 튜닝 {man['tune']}장 / 최종 평가 {man['eval']}장, seed `{man['seed']}`")
    w(f"- 아래 수치는 **평가 {man['eval']}장**에서만 산출했고, 이 분할은 개선 과정에 한 번도 사용하지 않았습니다.")
    w("- 이미지 파일은 커밋하지 않습니다. `python3 eval/fetch_data.py` 로 동일하게 재현됩니다.")
    w("")
    w("## 결과 (평가 70장, 동일 데이터·동일 지표)")
    w("")
    w("| 지표 | 변경 전 (legacy) | 변경 후 (v2) | 차이 |")
    w("| --- | ---: | ---: | ---: |")

    def row(label: str, key: str, pct: bool = True, better_up: bool = True):
        x, y = b[key], a[key]
        d = y - x
        arrow = "" if abs(d) < 1e-9 else (" ▲" if (d > 0) == better_up else " ▼")
        if pct:
            w(f"| {label} | {x*100:.1f}% | {y*100:.1f}% | {d*100:+.1f}p{arrow} |")
        else:
            w(f"| {label} | {x} | {y} | {d:+}{arrow} |")

    row("top-1 정확도", "top1_accuracy")
    row("top-5 정확도", "top5_accuracy")
    row("고양이 인식 성공률 (status=ok)", "cat_detected_rate")
    row("'고양이 없음' 오판율", "no_cat_rate", better_up=False)
    row("'야생 고양이과' 오판율", "wild_cat_rate", better_up=False)
    row("무늬 라벨이 1위인 비율", "pattern_label_rate", better_up=False)
    w(f"| 평균 추론 지연 | {b['mean_latency_ms']}ms | {a['mean_latency_ms']}ms | {a['mean_latency_ms']-b['mean_latency_ms']:+d}ms |")
    w("")
    w(f"- top-1: **{round(b['top1_accuracy']*70)}/70 → {round(a['top1_accuracy']*70)}/70**")
    w(f"- 새로 맞힌 이미지 {len(fixed)}건, 새로 틀린 이미지 {len(broke)}건")
    for f in fixed:
        w(f"  - 교정: `{f}` — {br[f]['top1']} → **{ar[f]['top1']}** (정답 {br[f]['expected']})")
    for f in broke:
        w(f"  - 퇴행: `{f}` — {br[f]['top1']} → {ar[f]['top1']} (정답 {br[f]['expected']})")
    w("")
    w("추론 지연이 2배로 늘어난 것은 두 모델 모두 뷰를 2개씩 보기 때문입니다. "
      "CPU 단일 요청 기준 여전히 1초 미만이고, 서비스 응답 시간에 체감될 수준은 아닙니다.")
    w("")
    w("## 품종별 top-1")
    w("")
    w("| 품종 | n | 변경 전 | 변경 후 |")
    w("| --- | ---: | ---: | ---: |")
    for breed in sorted(b["per_breed"]):
        x = b["per_breed"][breed]
        y = a["per_breed"][breed]
        mark = "" if abs(y["top1"] - x["top1"]) < 1e-9 else (" ▲" if y["top1"] > x["top1"] else " ▼")
        w(f"| {breed} | {x['n']} | {x['top1']*100:.0f}% | {y['top1']*100:.0f}%{mark} |")
    w("")
    w("## 개선 과정 (시도 2회, 튜닝 30장에서만 판단)")
    w("")
    w("| 시도 | 내용 | 튜닝 top-1 | 튜닝 '고양이 없음' | 채택 |")
    w("| --- | --- | ---: | ---: | --- |")
    bt = load("baseline", "tune")["summary"]
    t1 = load("attempt1_v1", "tune")["summary"]
    t2 = load("attempt2_v2", "tune")["summary"]
    w(f"| — | 변경 전 (legacy) | {bt['top1_accuracy']*100:.1f}% | {bt['no_cat_rate']*100:.1f}% | 기준 |")
    w(f"| 1 | 게이트 2뷰 **평균** + 품종 좌우반전 TTA | {t1['top1_accuracy']*100:.1f}% | {t1['no_cat_rate']*100:.1f}% | ✗ 악화 |")
    w(f"| 2 | 게이트 2뷰 **max** + 품종 2뷰 평균(늘림+패딩) | {t2['top1_accuracy']*100:.1f}% | {t2['no_cat_rate']*100:.1f}% | ✓ 채택 |")
    w("")
    path = ROOT / "eval" / "RESULTS.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
