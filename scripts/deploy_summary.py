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
    w("## 무중단 배포 검증 결과")
    w("")

    if not recs:
        w("> 모니터가 기록한 요청이 없습니다. 모니터 기동 자체가 실패했는지 확인하세요.")
        w("")
        out = "\n".join(L) + "\n"
        with open(args.summary_out, "w", encoding="utf-8") as f:
            f.write(out)
        print(out)
        return 0

    first, last = recs[0], recs[-1]
    span = last["end"]["epoch"] - first["start"]["epoch"]
    ok = [r for r in recs if r["analysis_ok"]]
    bad = [r for r in recs if not r["analysis_ok"]]

    w("| 항목 | 값 |")
    w("| --- | --- |")
    w(f"| 검증 주소 | `{meta.get('endpoint', args.url)}` |")
    w(f"| 검증 시간 (KST) | {first['start']['kst']} → {last['end']['kst']}  ({span:.1f}초) |")
    w(f"| 검증 시간 (UTC) | {first['start']['utc']} → {last['end']['utc']} |")
    w(f"| 고정 테스트 이미지 | `{meta.get('image', '-')}` (모든 요청 동일 바이트) |")
    w(f"| 요청 방식 | 응답 완료 후 {meta.get('interval_sec', 1)}초 대기 → 다음 요청 |")
    w(f"| 배포 결과 | **{args.deploy_result}** |")
    w("")

    w("### 구간별 요청 집계와 대표 응답")
    w("")
    w("| 구간 | 요청 | 정상 | 실패 | 대표 응답 |")
    w("| --- | ---: | ---: | ---: | --- |")
    for key, label in PHASES:
        ph = [r for r in recs if r["phase"] == key]
        if not ph:
            w(f"| {label} | 0 | 0 | 0 | — |")
            continue
        pok = [r for r in ph if r["analysis_ok"]]
        rep = pok[-1] if pok else ph[-1]
        w(f"| {label} | {len(ph)} | {len(pok)} | {len(ph) - len(pok)} | {brief(rep)} |")
    w(f"| **합계** | **{len(recs)}** | **{len(ok)}** | **{len(bad)}** | |")
    w("")

    # 버전 전환
    # 버전 문자열이 비어 있는 구버전 응답도 하나의 "버전"으로 취급해 전환을 감지한다.
    first_v = ok[0]["version"] if ok else ""
    first_label = first_v or "(버전 필드 없음)"
    switch = None
    for r in ok:
        if r["version"] != first_v:
            switch = r
            break
    w("### 버전 전환")
    w("")
    if switch:
        prev_ok = [r for r in ok if r["seq"] < switch["seq"]]
        w(f"- 전환 전 마지막 정상 응답: `#{prev_ok[-1]['seq']:04d}` {prev_ok[-1]['end']['kst']} → `{first_label}`")
        w(f"- 전환 후 첫 정상 응답: `#{switch['seq']:04d}` {switch['start']['kst']} → `{switch['version']}`")
        gap = switch["start"]["epoch"] - prev_ok[-1]["end"]["epoch"]
        w(f"- 두 응답 사이 **{gap:.1f}초**, 실패 {switch['seq'] - prev_ok[-1]['seq'] - 1}건")
    elif first_v:
        w(f"- 관측 구간 전체가 같은 버전입니다: `{first_v}` (버전 전환이 관측되지 않음)")
    elif ok:
        w("- 정상 응답은 있었지만 응답에 버전 정보가 없습니다. "
          "배포 전 이미지가 버전 필드를 넣기 전 빌드인 경우입니다.")
    else:
        w("- 정상 응답이 없어 버전을 확인하지 못했습니다.")
    w("")

    # 요청 간격 / 기록 공백
    starts = [r["start"]["epoch"] for r in recs]
    intervals = [b - a for a, b in zip(starts, starts[1:])]
    gaps = [recs[i + 1]["start"]["epoch"] - recs[i]["end"]["epoch"] for i in range(len(recs) - 1)]
    max_gap = max(gaps) if gaps else 0.0
    max_gap_i = gaps.index(max_gap) if gaps else -1

    w("### 요청 간격과 기록 공백")
    w("")
    w("| 항목 | 값 |")
    w("| --- | --- |")
    if intervals:
        w(f"| 실제 요청 간격 (시작→시작) | 평균 {statistics.mean(intervals):.2f}s · "
          f"중앙 {statistics.median(intervals):.2f}s · 최대 {max(intervals):.2f}s |")
    if gaps:
        g = recs[max_gap_i]
        w(f"| **최대 기록 공백** (이전 응답 종료→다음 요청 시작) | **{max_gap:.2f}s** "
          f"(`#{g['seq']:04d}` {g['end']['kst']} 이후) |")
    w(f"| 관측 밀도 | {span / max(len(recs), 1):.2f}초당 1건 |")
    w("")
    w(f"> 기록 공백 구간은 **관측하지 않은 구간**입니다. 그 사이의 서비스 상태는 이 검증으로 "
      f"확인되지 않았으므로, 아래 무중단 판단은 최대 {max_gap:.2f}초의 미검증 구간을 전제로 합니다.")
    w("")

    # 무중단 판단
    w("### 관측 범위에서의 무중단 여부")
    w("")
    runs = []
    cur = []
    for r in recs:
        if r["analysis_ok"]:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)

    if not bad:
        w(f"- 관측한 **{len(recs)}건 전부 정상**이었습니다. "
          f"{first['start']['kst']}부터 {last['end']['kst']}까지, 최대 {max_gap:.2f}초의 "
          f"미검증 구간을 제외하면 **중단이 관측되지 않았습니다.**")
    else:
        longest = max(runs, key=len)
        s, e = longest[0], longest[-1]
        before_ok = [r for r in ok if r["seq"] < s["seq"]]
        after_ok = [r for r in ok if r["seq"] > e["seq"]]
        lo = before_ok[-1]["end"]["epoch"] if before_ok else s["start"]["epoch"]
        hi = after_ok[0]["start"]["epoch"] if after_ok else e["end"]["epoch"]
        w(f"- 실패 {len(bad)}건이 관측되었습니다. 가장 긴 연속 실패는 "
          f"`#{s['seq']:04d}`~`#{e['seq']:04d}` ({len(longest)}건), "
          f"{s['start']['kst']} ~ {e['end']['kst']} 입니다.")
        w(f"- 마지막 정상 응답과 다음 정상 응답 사이는 **{hi - lo:.1f}초** 입니다. "
          f"이 구간에서 서비스는 **중단되었습니다** (무중단 아님).")
        w(f"- 컨테이너를 내리고 새로 띄우는 교체 방식이므로 교체 시간만큼 공백이 생깁니다. "
          f"실제 무중단을 위해서는 앞단 프록시를 두고 신·구 컨테이너를 겹쳐 띄우는 방식이 필요합니다.")
    w("")
    w(f"<sub>요청 {len(recs)}건의 개별 기록은 Actions 로그와 Artifacts(`monitor.jsonl`)에 있습니다.</sub>")

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
