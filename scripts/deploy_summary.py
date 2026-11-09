"""모니터 로그(JSON Lines)를 읽어 Actions 로그와 Step Summary 를 만든다.

- 요청별 결과는 Actions 로그에 한 줄씩 출력한다.
- Summary 는 스크린샷 한 장에 담기도록 표 위주로 짧게 구성한다.
표준 라이브러리만 사용한다.

실행: python3 scripts/deploy_summary.py --log logs/monitor.jsonl --summary-out logs/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

PHASES = [("before", "배포 전"), ("during", "배포 중"), ("after", "배포 후")]


def fmt(rec: dict) -> str:
    mark = "OK  " if rec["analysis_ok"] else "FAIL"
    return (
        f"#{rec['seq']:04d} {rec['phase']:6s} {mark} "
        f"{rec['start']['kst']}->{rec['end']['kst']} {rec['response_ms']:5d}ms "
        f"http={rec['http_status']:3d} 분석={'정상' if rec['analysis_ok'] else '비정상'} "
        f"버전={rec['version'] or '-'} {rec['result_status']} {rec['error']}"
    )


def brief(rec: dict) -> str:
    if not rec:
        return "—"
    return (
        f"`#{rec['seq']:04d}` {rec['start']['kst']} · "
        f"{'정상' if rec['analysis_ok'] else '실패'} · {rec['response_ms']}ms · "
        f"HTTP {rec['http_status']} · `{rec['version'] or '-'}`"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--deploy-result", default="unknown")
    ap.add_argument("--image-tag", default="")
    args = ap.parse_args()

    meta: dict = {}
    recs: list = []
    if os.path.exists(args.log):
        with open(args.log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") == "meta":
                    meta = d
                elif d.get("type") == "request":
                    recs.append(d)

    print("=" * 100)
    print("요청별 결과 (고정 테스트 이미지, 응답 완료 후 1초 간격)")
    print("=" * 100)
    for r in recs:
        print(fmt(r))
    print("=" * 100)

    L: list = []
    w = L.append

    if not recs:
        w("## 무중단 배포 검증 — 기록 없음")
        w("")
        w("> 모니터가 기록한 요청이 없습니다. 모니터 기동 자체가 실패했는지 확인하세요.")
        out = "\n".join(L) + "\n"
        with open(args.summary_out, "w", encoding="utf-8") as f:
            f.write(out)
        gh = os.environ.get("GITHUB_STEP_SUMMARY")
        if gh:
            with open(gh, "a", encoding="utf-8") as f:
                f.write(out)
        print(out)
        return 0

    first, last = recs[0], recs[-1]
    span = last["end"]["epoch"] - first["start"]["epoch"]
    ok = [r for r in recs if r["analysis_ok"]]
    bad = [r for r in recs if not r["analysis_ok"]]

    starts = [r["start"]["epoch"] for r in recs]
    intervals = [b - a for a, b in zip(starts, starts[1:])]
    gaps = [recs[i + 1]["start"]["epoch"] - recs[i]["end"]["epoch"] for i in range(len(recs) - 1)]
    max_gap = max(gaps) if gaps else 0.0
    gap_at = recs[gaps.index(max_gap)] if gaps else None

    # 버전 전환 (버전 필드가 없는 구버전 응답도 하나의 버전으로 취급)
    first_v = ok[0]["version"] if ok else ""
    first_label = first_v or "(버전 필드 없음)"
    switch = next((r for r in ok if r["version"] != first_v), None)

    # 제출용 스크린샷 한 장에 들어가도록, 표 3개 + 결론 한 문단으로만 구성한다.
    w(f"## 무중단 배포 검증 — {args.deploy_result}")
    w("")
    w("| 검증 주소 | 검증 시간 (KST) | 고정 테스트 이미지 | 요청 방식 | 배포 이미지 |")
    w("| --- | --- | --- | --- | --- |")
    w(f"| `{meta.get('endpoint', args.url)}` "
      f"| {first['start']['kst'][:8]} → {last['end']['kst'][:8]} ({span:.1f}초) "
      f"| `{meta.get('image', '-')}` (전 요청 동일 바이트) "
      f"| 응답 완료 후 {meta.get('interval_sec', 1)}초 "
      f"| `{args.image_tag or '-'}` |")
    w("")
    w("| 구간 | 요청 | 정상 | 실패 | 대표 응답 — 시각 · 응답시간 · 버전 |")
    w("| --- | ---: | ---: | ---: | --- |")
    for key, label in PHASES:
        ph = [r for r in recs if r["phase"] == key]
        if not ph:
            w(f"| {label} | 0 | 0 | 0 | — |")
            continue
        pok = [r for r in ph if r["analysis_ok"]]
        rep = pok[-1] if pok else ph[-1]
        w(f"| {label} | {len(ph)} | {len(pok)} | {len(ph) - len(pok)} "
          f"| `#{rep['seq']:04d}` {rep['start']['kst'][:8]} · {rep['response_ms']}ms · "
          f"`{rep['version'] or '(버전 필드 없음)'}` |")
    if switch:
        prev_ok = [r for r in ok if r["seq"] < switch["seq"]][-1]
        sw = (f"**버전 전환** `#{prev_ok['seq']:04d}` `{first_label}` → "
              f"`#{switch['seq']:04d}` `{switch['version']}` "
              f"({switch['start']['epoch'] - prev_ok['end']['epoch']:.1f}초, "
              f"실패 {switch['seq'] - prev_ok['seq'] - 1}건)")
    elif first_v:
        sw = f"**버전 전환 없음** — 전 구간 `{first_v}`"
    else:
        sw = "**버전 전환 확인 불가**"
    w(f"| **합계** | **{len(recs)}** | **{len(ok)}** | **{len(bad)}** | {sw} |")
    w("")
    w("| 실제 요청 간격 (시작→시작) | 최대 기록 공백 = 미검증 구간 | 관측 밀도 |")
    w("| --- | --- | --- |")
    iv = (f"평균 {statistics.mean(intervals):.2f}s · 중앙 {statistics.median(intervals):.2f}s · "
          f"최대 {max(intervals):.2f}s") if intervals else "—"
    gp = f"**{max_gap:.2f}s** (`#{gap_at['seq']:04d}` 응답 종료 후)" if gap_at else "—"
    w(f"| {iv} | {gp} | {span / max(len(recs), 1):.2f}초당 1건 |")
    w("")

    # 결론 한 문단
    runs, cur = [], []
    for r in recs:
        if r["analysis_ok"]:
            if cur:
                runs.append(cur); cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)

    if not bad:
        w(f"**무중단 여부 — 관측 범위에서 중단 없음.** 요청 {len(recs)}건 전부 정상이었습니다. "
          f"다만 요청 사이 최대 {max_gap:.2f}초는 관측하지 않은 구간이라 그 사이 상태는 검증되지 않았습니다.")
    else:
        longest = max(runs, key=len)
        s0, e0 = longest[0], longest[-1]
        before_ok = [r for r in ok if r["seq"] < s0["seq"]]
        after_ok = [r for r in ok if r["seq"] > e0["seq"]]
        lo = before_ok[-1]["end"]["epoch"] if before_ok else s0["start"]["epoch"]
        hi = after_ok[0]["start"]["epoch"] if after_ok else e0["end"]["epoch"]
        w(f"**무중단 여부 — 아님.** 요청 {len(recs)}건 중 실패 {len(bad)}건이 관측됐습니다 "
          f"(최장 연속 실패 `#{s0['seq']:04d}`~`#{e0['seq']:04d}`, "
          f"{s0['start']['kst'][:8]}~{e0['end']['kst'][:8]}). "
          f"마지막 정상 응답과 다음 정상 응답 사이 **{hi - lo:.1f}초 동안 서비스가 중단**됐습니다. "
          f"컨테이너를 내렸다 새로 띄우는 교체 방식이라 교체 시간만큼 공백이 생깁니다 — "
          f"실제 무중단에는 앞단 프록시를 두고 신·구 컨테이너를 겹쳐 띄우는 방식이 필요합니다. "
          f"요청 사이 최대 {max_gap:.2f}초는 관측하지 않은 구간이라, 위 판단은 그만큼의 "
          f"미검증 구간을 전제로 합니다.")
    w("")
    w(f"<sub>구간 라벨은 요청 시작 시점 기준이라 경계에서 1건이 걸칠 수 있습니다. "
      f"요청 {len(recs)}건의 개별 기록은 Actions 로그와 Artifacts(`monitor.jsonl`, UTF-8)에 있습니다.</sub>")

    out = "\n".join(L) + "\n"
    with open(args.summary_out, "w", encoding="utf-8") as f:
        f.write(out)
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
