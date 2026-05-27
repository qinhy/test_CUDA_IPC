#!/usr/bin/env python3
import time
import struct
from multiprocessing.shared_memory import SharedMemory

import cupy as cp


# Must match viewer
SHM_NAME = "cuda_ipc_rgba_stream_demo_v1"

WIDTH = 640
HEIGHT = 480
CHANNELS = 4
NUM_SLOTS = 4
FPS = 60

MAGIC = b"CIPCRGBA"  # 8 bytes

HEADER_FMT = "<8sIiiiiQQ"
# magic:8s
# handle_len:uint32
# width:int32
# height:int32
# num_slots:int32
# channels:int32
# itemsize:uint64
# frame_bytes:uint64

HEADER_SIZE = struct.calcsize(HEADER_FMT)

HANDLE_OFFSET = 256
CTRL_OFFSET = 512

LATEST_OFFSET = CTRL_OFFSET          # int64
STOP_OFFSET = CTRL_OFFSET + 8        # int64
GEN_OFFSET = CTRL_OFFSET + 16        # int64[NUM_SLOTS]

SHM_SIZE = 4096


def write_i64(buf, offset, value):
    struct.pack_into("<q", buf, offset, int(value))


def read_i64(buf, offset):
    return struct.unpack_from("<q", buf, offset)[0]


def gen_offset(slot):
    return GEN_OFFSET + slot * 8


def create_clean_shm(name, size):
    # Remove stale shm from previous crashed run if possible.
    try:
        old = SharedMemory(name=name, create=False)
        old.close()
        old.unlink()
        print(f"[producer] removed stale shm: {name}", flush=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[producer] could not remove old shm, maybe still in use: {e}", flush=True)

    return SharedMemory(name=name, create=True, size=size)


def fill_rgba_frame(frames, slot, x, y, seq):
    # frames shape: (NUM_SLOTS, HEIGHT, WIDTH, 4), dtype=uint8
    # Make a moving color pattern.
    frames[slot, :, :, 0] = ((x + seq * 3) & 255).astype(cp.uint8)          # R
    frames[slot, :, :, 1] = ((y + seq * 2) & 255).astype(cp.uint8)          # G
    frames[slot, :, :, 2] = (((x ^ y) + seq * 5) & 255).astype(cp.uint8)    # B
    frames[slot, :, :, 3] = cp.uint8(255)                                  # A


def main():
    cp.cuda.Device(0).use()

    frame_bytes = WIDTH * HEIGHT * CHANNELS  # uint8 RGBA
    itemsize = 1

    shm = create_clean_shm(SHM_NAME, SHM_SIZE)
    buf = shm.buf

    # GPU ring buffer.
    frames = cp.empty((NUM_SLOTS, HEIGHT, WIDTH, CHANNELS), dtype=cp.uint8)

    # Precompute coordinate arrays on GPU.
    x = cp.arange(WIDTH, dtype=cp.uint16)[None, :]
    y = cp.arange(HEIGHT, dtype=cp.uint16)[:, None]

    # Export the base allocation.
    handle = cp.cuda.runtime.ipcGetMemHandle(frames.data.ptr)

    if len(handle) + HANDLE_OFFSET > CTRL_OFFSET:
        raise RuntimeError("CUDA IPC handle does not fit in shm layout")

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        len(handle),
        WIDTH,
        HEIGHT,
        NUM_SLOTS,
        CHANNELS,
        itemsize,
        frame_bytes,
    )

    buf[:HEADER_SIZE] = header
    buf[HANDLE_OFFSET: HANDLE_OFFSET + len(handle)] = handle

    write_i64(buf, LATEST_OFFSET, -1)
    write_i64(buf, STOP_OFFSET, 0)

    for s in range(NUM_SLOTS):
        write_i64(buf, gen_offset(s), 0)

    print("[producer] shm name:", SHM_NAME, flush=True)
    print("[producer] gpu ptr :", hex(frames.data.ptr), flush=True)
    print("[producer] handle bytes:", len(handle), flush=True)
    print("[producer] frame:", WIDTH, "x", HEIGHT, "RGBA8", flush=True)
    print("[producer] slots:", NUM_SLOTS, flush=True)
    print("[producer] streaming. Ctrl+C to stop.", flush=True)

    seq = 0
    delay = 1.0 / FPS

    try:
        while True:
            slot = seq % NUM_SLOTS

            # Odd generation = producer is writing this slot.
            write_i64(buf, gen_offset(slot), seq * 2 + 1)

            fill_rgba_frame(frames, slot, x, y, seq)

            # Ensure GPU writes are complete before publishing.
            cp.cuda.runtime.deviceSynchronize()

            # Even generation = stable.
            write_i64(buf, gen_offset(slot), seq * 2 + 2)

            # Publish latest complete frame.
            write_i64(buf, LATEST_OFFSET, seq)

            if seq % FPS == 0:
                print(f"[producer] published seq={seq} slot={slot}", flush=True)

            seq += 1
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n[producer] stopping", flush=True)

    finally:
        # Signal viewer no more frames.
        try:
            write_i64(buf, STOP_OFFSET, 1)
        except Exception:
            pass

        # Keep allocation alive briefly so viewer can see STOP.
        time.sleep(0.5)

        shm.close()
        shm.unlink()
        print("[producer] cleaned shm and exiting", flush=True)


if __name__ == "__main__":
    main()