"""Generate PWA icons from the repository favicon geometry (no external assets)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / "public"
BG = (15, 20, 25, 255)  # #0f1419
ACCENT = (59, 130, 246, 255)  # #3b82f6


def _inside_rounded_rect(x: float, y: float, size: float, radius: float) -> bool:
    if x < 0 or y < 0 or x > size or y > size:
        return False
    r = radius
    if x < r and y < r:
        return (x - r) ** 2 + (y - r) ** 2 <= r**2
    if x > size - r and y < r:
        return (x - (size - r)) ** 2 + (y - r) ** 2 <= r**2
    if x < r and y > size - r:
        return (x - r) ** 2 + (y - (size - r)) ** 2 <= r**2
    if x > size - r and y > size - r:
        return (x - (size - r)) ** 2 + (y - (size - r)) ** 2 <= r**2
    return True


def _inside_f_mark(x: float, y: float) -> bool:
    """Approximate favicon.svg path in 64x64 viewBox coordinates."""
    if 16 <= x <= 23 and 22 <= y <= 42:
        return True
    if 16 <= x <= 32 and 22 <= y <= 29:
        return True
    if 16 <= x <= 30 and 30 <= y <= 36:
        return True
    if 23 <= x <= 39 and 36 <= y <= 42:
        return True
    if 39 <= x <= 47 and 22 <= y <= 28:
        return True
    if 32 <= x <= 48 and 28 <= y <= 42:
        return True
    return False


def _pixel_rgba(x: int, y: int, size: int, *, maskable: bool) -> tuple[int, int, int, int]:
    if maskable:
        margin = int(size * 0.1)
        if x < margin or y < margin or x >= size - margin or y >= size - margin:
            return BG

    vx = (x + 0.5) / size * 64.0
    vy = (y + 0.5) / size * 64.0
    if not _inside_rounded_rect(vx, vy, 64.0, 12.0):
        return (0, 0, 0, 0)
    if _inside_f_mark(vx, vy):
        return ACCENT
    return BG


def write_png(size: int, path: Path, *, maskable: bool = False) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            row.extend(_pixel_rgba(x, y, size, maskable=maskable))
        rows.append(bytes(row))
    compressed = zlib.compress(b"".join(rows), 9)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    write_png(192, PUBLIC / "pwa-192x192.png")
    write_png(512, PUBLIC / "pwa-512x512.png")
    write_png(192, PUBLIC / "pwa-192x192-maskable.png", maskable=True)
    write_png(512, PUBLIC / "pwa-512x512-maskable.png", maskable=True)
    write_png(180, PUBLIC / "apple-touch-icon.png", maskable=True)


if __name__ == "__main__":
    main()
