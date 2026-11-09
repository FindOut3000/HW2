"""배포 검증에 쓰는 고정 테스트 이미지를 만든다.

바이너리를 저장소에 넣지 않고 코드로 생성한다. 같은 코드는 항상 같은 바이트를 만들므로
배포 전·중·후 요청이 모두 동일한 입력을 쓴다는 점이 보장된다. 표준 라이브러리만 쓴다.

실행: python3 scripts/probe_image.py <출력경로>
"""
from __future__ import annotations

import struct
import sys
import zlib

WIDTH = HEIGHT = 224


def build_png() -> bytes:
    rows = []
    for y in range(HEIGHT):
        row = bytearray(b"\x00")  # 필터 타입 0
        for x in range(WIDTH):
            # 좌표만으로 결정되는 고정 패턴 (난수 없음)
            row += bytes(((x * 255) // WIDTH, (y * 255) // HEIGHT, 160))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("사용법: python3 scripts/probe_image.py <출력경로>", file=sys.stderr)
        return 2
    data = build_png()
    with open(sys.argv[1], "wb") as f:
        f.write(data)
    print(f"고정 테스트 이미지 생성: {sys.argv[1]} ({len(data)} bytes, sha1={zlib.crc32(data):08x})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
