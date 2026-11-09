"""평가용 분할에 대해 현재 추론 파이프라인의 정확도를 측정한다.

운영 코드와 다른 경로를 타지 않도록 app.py 의 open_image / ModelBundle 을 그대로 쓴다.

실행:
  python3 eval/run_eval.py --split tune  --name baseline
  python3 eval/run_eval.py --split eval  --name baseline
환경변수 PREPROCESS_MODE 로 전처리 방식을 고른다 (app.py 참조).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "vendor").is_dir():
    sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

DATA = ROOT / "eval" / "data"
OUT = ROOT / "eval" / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "eval", "all"], required=True)
    ap.add_argument("--name", required=True, help="결과 파일 이름")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import app  # noqa: E402  (vendor 경로 설정 후에 import)

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    items = [i for i in manifest["items"] if args.split == "all" or i["split"] == args.split]
    if args.limit:
        items = items[: args.limit]

    app.MODELS.load()
    if not app.MODELS.ready:
        print(f"모델 로드 실패: {app.MODELS.error}", file=sys.stderr)
        return 1

    mode = os.environ.get("PREPROCESS_MODE", "(기본값)")
    print(f"분할={args.split}  장수={len(items)}  PREPROCESS_MODE={mode}")

    records = []
    t_all = time.perf_counter()
    for n, it in enumerate(items, 1):
        raw = (DATA / "images" / it["file"]).read_bytes()
        image = app.open_image(raw)          # 운영과 동일한 디코딩/EXIF 보정
        t0 = time.perf_counter()
        r = app.MODELS.predict(image)
        ms = int((time.perf_counter() - t0) * 1000)

        preds = r["predictions"]
        top5 = [p["breed"] for p in preds]
        expected = it["expected_breed"]
        domestic, wild = r["domestic_cat_score"], r["wild_feline_score"]
        if domestic < app.CAT_PRESENT_THRESHOLD and wild < app.CAT_PRESENT_THRESHOLD:
            status = "no_cat"
        elif wild >= app.WILD_THRESHOLD and wild > domestic:
            status = "wild_cat"
        else:
            status = "ok"

        records.append({
            "file": it["file"],
            "expected": expected,
            "top1": top5[0],
            "top1_prob": preds[0]["probability"],
            "top1_is_pattern": preds[0]["is_pattern"],
            "top5": top5,
            "correct_top1": top5[0] == expected,
            "correct_top5": expected in top5,
            "status": status,
            "domestic_cat_score": domestic,
            "wild_feline_score": wild,
            "elapsed_ms": ms,
        })
        if n % 10 == 0 or n == len(items):
            acc = sum(x["correct_top1"] for x in records) / len(records)
            print(f"  {n:3d}/{len(items)}  top1={acc:.3f}")

    total_s = time.perf_counter() - t_all
    n = len(records)
    top1 = sum(r["correct_top1"] for r in records)
    top5 = sum(r["correct_top5"] for r in records)

    per_breed: dict[str, dict] = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})
    for r in records:
        b = per_breed[r["expected"]]
        b["n"] += 1
        b["top1"] += r["correct_top1"]
        b["top5"] += r["correct_top5"]

    summary = {
        "name": args.name,
        "split": args.split,
        "preprocess_mode": mode,
        "n": n,
        "top1_accuracy": round(top1 / n, 4),
        "top5_accuracy": round(top5 / n, 4),
        "cat_detected_rate": round(sum(r["status"] == "ok" for r in records) / n, 4),
        "no_cat_rate": round(sum(r["status"] == "no_cat" for r in records) / n, 4),
        "wild_cat_rate": round(sum(r["status"] == "wild_cat" for r in records) / n, 4),
        "pattern_label_rate": round(sum(r["top1_is_pattern"] for r in records) / n, 4),
        "mean_top1_prob": round(sum(r["top1_prob"] for r in records) / n, 4),
        "mean_latency_ms": int(sum(r["elapsed_ms"] for r in records) / n),
        "total_seconds": round(total_s, 1),
        "per_breed": {
            k: {"n": v["n"], "top1": round(v["top1"] / v["n"], 4), "top5": round(v["top5"] / v["n"], 4)}
            for k, v in sorted(per_breed.items())
        },
        "top_confusions": Counter(
            f'{r["expected"]} -> {r["top1"]}' for r in records if not r["correct_top1"]
        ).most_common(12),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name}__{args.split}.json"
    path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[{args.name} / {args.split}]  n={n}")
    print(f"  top-1 정확도      {summary['top1_accuracy']:.3f}  ({top1}/{n})")
    print(f"  top-5 정확도      {summary['top5_accuracy']:.3f}  ({top5}/{n})")
    print(f"  고양이 인식(ok)   {summary['cat_detected_rate']:.3f}   no_cat {summary['no_cat_rate']:.3f}   wild {summary['wild_cat_rate']:.3f}")
    print(f"  무늬 라벨 1위 비율 {summary['pattern_label_rate']:.3f}")
    print(f"  평균 top1 확률     {summary['mean_top1_prob']:.3f}   평균 지연 {summary['mean_latency_ms']}ms")
    print(f"  저장: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
