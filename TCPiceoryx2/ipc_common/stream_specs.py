from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamSpec:
    name: str
    port: int
    codec_hint: str
    is_mono: bool = False


STREAMS: dict[str, StreamSpec] = {
    "rgb": StreamSpec("rgb", 5000, "hevc", False),
    "left": StreamSpec("left", 5001, "h264", True),
    "right": StreamSpec("right", 5002, "h264", True),
}


def parse_stream_names(streams_arg: str) -> list[str]:
    names = [s.strip() for s in streams_arg.split(",") if s.strip()]
    if not names:
        raise RuntimeError("No streams selected")

    for name in names:
        if name not in STREAMS:
            raise RuntimeError(f"Unknown stream '{name}'. Valid: {','.join(STREAMS)}")

    return names
