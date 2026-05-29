#!/usr/bin/env python3
"""OAK-D PoE-like simulator -> iceoryx2 SHM publisher.

Publishes complete encoded access units, equivalent to DepthAI EncodedFrame-ish
messages, not arbitrary TCP bytes.
"""
from __future__ import annotations

import argparse
import time

import iceoryx2 as iox2

from iox_video_types import EncodedVideoSample, codec_from_name
from oakd_sim_core import OakdPoeLikeDevice, add_common_args, parse_stream_specs


def copy_payload(dst_sample, payload: bytes) -> None:
    if len(payload) > len(dst_sample.payload):
        raise RuntimeError(f"encoded packet too large: {len(payload)} > {len(dst_sample.payload)}")
    dst_sample.payload_size = len(payload)
    dst_sample.payload[: len(payload)] = payload


def main() -> int:
    p = argparse.ArgumentParser(description="OAK-D-PoE-like encoded simulator to iceoryx2 SHM")
    add_common_args(p)
    p.add_argument("--service-prefix", default="sim/video", help="services become <prefix>/<stream>/Encoded")
    args = p.parse_args()

    specs = parse_stream_specs(args.streams)
    iox2.set_log_level_from_env_or(iox2.LogLevel.Info)
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

    publishers = {}
    for spec in specs:
        service_name = f"{args.service_prefix}/{spec.name}"
        service = (
            node.service_builder(iox2.ServiceName.new(service_name))
            .publish_subscribe(EncodedVideoSample)
            .open_or_create()
        )
        publishers[spec.name] = service.publisher_builder().create()
        print(f"[{spec.name}] publishing {spec.codec} to iceoryx2 service '{service_name}'")

    with OakdPoeLikeDevice(
        specs=specs,
        fps=args.fps,
        keyframe_seconds=args.keyframe_seconds,
        ffmpeg_bin=args.ffmpeg_bin,
        encoder_preset=args.encoder_preset,
        ffmpeg_loglevel=args.ffmpeg_loglevel,
        queue_size=args.queue_size,
        video_file=args.video_file,
        loop_video=not args.no_loop_video,
    ) as dev:
        queues = {s.name: dev.getOutputQueue(s.name, blocking=False) for s in specs}
        last_report = time.monotonic()
        counts = {s.name: 0 for s in specs}
        while True:
            any_pkt = False
            for spec in specs:
                pkt = queues[spec.name].tryGet()
                if pkt is None:
                    continue
                any_pkt = True
                sample = publishers[spec.name].loan_uninit()

                # Python iceoryx2 binding: SampleMutUninit exposes payload_ptr,
                # and the extension method payload() casts it to POINTER(EncodedVideoSample).
                payload = sample.payload().contents

                payload.sequence_number = pkt.sequence_number
                payload.timestamp_ns = pkt.timestamp_ns
                payload.codec = codec_from_name(pkt.codec)
                payload.width = pkt.width
                payload.height = pkt.height
                payload.fps_num = int(round(pkt.fps * 1000.0))
                payload.fps_den = 1000
                payload.is_keyframe = 1 if pkt.is_keyframe else 0
                payload.has_headers = 1 if pkt.has_headers else 0
                payload.stream_id = pkt.stream_id
                copy_payload(payload, pkt.payload)

                sample.assume_init().send()
                counts[spec.name] += 1
            if not any_pkt:
                time.sleep(0.001)
            now = time.monotonic()
            if now - last_report >= 10.0:
                print("published " + ", ".join(f"{k}={v}" for k, v in counts.items()) + " packets/10s")
                counts = {s.name: 0 for s in specs}
                last_report = now


if __name__ == "__main__":
    raise SystemExit(main())
