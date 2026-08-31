from pathlib import Path
import struct
import zlib

PUBLIC = Path(__file__).resolve().parents[1] / "public"


def write_png(size: int, path: Path) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    row = b"\x00" + bytes([15, 20, 25, 255] * size)
    raw = row * size
    compressed = zlib.compress(raw, 9)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for size in (192, 512):
        write_png(size, PUBLIC / f"pwa-{size}x{size}.png")


if __name__ == "__main__":
    main()
