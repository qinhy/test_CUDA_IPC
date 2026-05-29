#!/usr/bin/env python3
"""ctypes payloads for iceoryx2 video samples.

iceoryx2 Python publish/subscribe examples use fixed-size ctypes.Structure
payloads. Encoded video access units are variable size, so this demo uses a
fixed maximum payload and a payload_size field.
"""
from __future__ import annotations

import ctypes

CODEC_H264 = 1
CODEC_HEVC = 2
DEFAULT_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024


class EncodedVideoSample(ctypes.Structure):
    _fields_ = [
        ("sequence_number", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("codec", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("fps_num", ctypes.c_uint32),
        ("fps_den", ctypes.c_uint32),
        ("payload_size", ctypes.c_uint32),
        ("is_keyframe", ctypes.c_uint8),
        ("has_headers", ctypes.c_uint8),
        ("stream_id", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8),
        ("reserved1", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * DEFAULT_MAX_PAYLOAD_BYTES),
    ]

    @staticmethod
    def type_name() -> str:
        return "EncodedVideoSample.v1.max4MiB"


def codec_name(codec: int) -> str:
    if codec == CODEC_H264:
        return "h264"
    if codec == CODEC_HEVC:
        return "hevc"
    return f"unknown({codec})"


def codec_from_name(name: str) -> int:
    n = name.lower()
    if n in ("h264", "avc"):
        return CODEC_H264
    if n in ("hevc", "h265"):
        return CODEC_HEVC
    raise ValueError(f"unsupported codec: {name}")
