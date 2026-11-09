"""배포 중 서비스 가용성을 재는 독립 모니터.

배포 단계와 분리된 프로세스로 돌면서, 고정 테스트 이미지를 실제 분석 API 에 보내고
응답 하나가 끝나면 지정한 간격(기본 1초)을 쉬었다가 다음 요청을 보낸다.
요청 하나당 한 줄의 JSON 을 UTF-8 로 기록한다 (JSON Lines).

기록 항목: 순번, 단계(before/during/after), 시작·종료 시각(UTC/KST), 응답 시간(ms),
HTTP 상태, 분석 정상 여부, 그 응답을 돌려준 배포 버전.

표준 라이브러리만 사용한다 (러너의 /usr/bin/python3 로 바로 실행 가능).

실행 예:
  python3 scripts/deploy_monitor.py \
      --url http://127.0.0.1:8000 --image probe.png \
      --out logs/monitor.jsonl --phase-file logs/phase --stop-file logs/stop
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request

KST = dt.timezone(dt.timedelta(hours=9))
BOUNDARY = "----catbreeddeploymonitor"


def stamps(epoch: float) -> dict:
    u = dt.datetime.fromtimestamp(epoch, dt.timezone.utc)
    return {
        "utc": u.strftime("%Y-%m-%dT%H:%M:%S.") + f"{u.microsecond // 1000:03d}Z",
        "kst": u.astimezone(KST).strftime("%H:%M:%S.") + f"{u.microsecond // 1000:03d}",
        "epoch": round(epoch, 3),
    }


def build_body(image: bytes) -> bytes:
    head = (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="probe.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8")
    return head + image + f"\r\n--{BOUNDARY}--\r\n".encode("utf-8")


def one_request(url: str, body: bytes, timeout: float) -> dict:
    """요청 1회. 어떤 경우에도 예외를 밖으로 내보내지 않는다."""
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        method="POST",
    )
    t0 = time.time()
    http_status = 0
    payload = None
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            http_status = r.status
            payload = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = None
        error = f"HTTP {exc.code}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    t1 = time.time()

    analysis_ok = False
    version = ""
    status = ""
    top = ""
    if isinstance(payload, dict):
        status = str(payload.get("status", ""))
        version = str((payload.get("version") or {}).get("build", ""))
        preds = payload.get("predictions") or []
        analysis_ok = (
            http_status == 200
            and payload.get("is_estimate") is True
            and len(preds) == 5
            and status in {"ok", "no_cat", "wild_cat"}
        )
        if preds:
            top = f'{preds[0].get("breed", "")} {preds[0].get("probability", "")}'

    return {
        "start": stamps(t0),
        "end": stamps(t1),
        "response_ms": int((t1 - t0) * 1000),
        "http_status": http_status,
        "ok": http_status == 200 and not error,
        "analysis_ok": analysis_ok,
        "version": version,
        "result_status": status,
        "top1": top,
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="서비스 기준 주소 (예: http://127.0.0.1:8000)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase-file", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--interval", type=float, default=1.0, help="응답 완료 후 대기 시간(초)")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--max-seconds", type=float, default=2400.0, help="안전 상한")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        body = build_body(f.read())

    endpoint = args.url.rstrip("/") + "/api/analyze"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    started = time.time()
    seq = 0
    # line-buffered + UTF-8 로 고정해 중간에 죽어도 기록이 남게 한다.
    with open(args.out, "a", encoding="utf-8", buffering=1) as log:
        meta = {
            "type": "meta",
            "endpoint": endpoint,
            "interval_sec": args.interval,
            "timeout_sec": args.timeout,
            "image": os.path.basename(args.image),
            "monitor_started": stamps(started),
        }
        log.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"[monitor] 시작 endpoint={endpoint} interval={args.interval}s", flush=True)

        while True:
            if os.path.exists(args.stop_file):
                print("[monitor] stop 파일 감지 — 종료", flush=True)
                break
            if time.time() - started > args.max_seconds:
                print("[monitor] 상한 도달 — 종료", flush=True)
                break

            phase = "unknown"
            try:
                with open(args.phase_file, encoding="utf-8") as pf:
                    phase = pf.read().strip() or "unknown"
            except OSError:
                pass

            seq += 1
            rec = one_request(endpoint, body, args.timeout)
            rec.update({"type": "request", "seq": seq, "phase": phase})
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mark = "OK " if rec["analysis_ok"] else "FAIL"
            print(
                f"[monitor] #{seq:04d} {phase:6s} {mark} "
                f"{rec['start']['kst']}->{rec['end']['kst']} "
                f"{rec['response_ms']:5d}ms http={rec['http_status']} "
                f"ver={rec['version'] or '-'} {rec['error']}",
                flush=True,
            )
            time.sleep(args.interval)

    print(f"[monitor] 총 {seq}건 기록 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
