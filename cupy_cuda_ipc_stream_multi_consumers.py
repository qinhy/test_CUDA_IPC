import time
import struct
from multiprocessing import Process, Event, set_start_method
from multiprocessing.shared_memory import SharedMemory

# ----------------------------
# Config
# ----------------------------
N = 16
NUM_SLOTS = 4
NUM_FRAMES = 30
NUM_CONSUMERS = 3

PRODUCER_SLEEP_SEC = 0.05
CONSUMER_SLEEP_SEC = 0.03

# ----------------------------
# Host shm layout
# ----------------------------
HEADER_FMT = "IQQI"  # handle_len:uint32, n:uint64, itemsize:uint64, num_slots:uint32
HEADER_SIZE = struct.calcsize(HEADER_FMT)

HANDLE_OFFSET = 128
CTRL_OFFSET = 512

LATEST_OFFSET = CTRL_OFFSET          # int64 latest seq
STOP_OFFSET = CTRL_OFFSET + 8        # int64 0/1
GEN_OFFSET = CTRL_OFFSET + 16        # int64[NUM_SLOTS]

SHM_SIZE = 4096


def write_i64(buf, offset, value):
    struct.pack_into("q", buf, offset, value)


def read_i64(buf, offset):
    return struct.unpack_from("q", buf, offset)[0]


def gen_offset(slot):
    return GEN_OFFSET + slot * 8


def producer(shm_name, ready, consumer_done_events):
    import cupy as cp

    cp.cuda.Device(0).use()

    # Ring buffer on GPU.
    frames = cp.empty((NUM_SLOTS, N), dtype=cp.float32)
    base = cp.arange(N, dtype=cp.float32)

    # Export the whole GPU allocation once.
    handle = cp.cuda.runtime.ipcGetMemHandle(frames.data.ptr)

    shm = SharedMemory(name=shm_name)
    buf = shm.buf

    try:
        header = struct.pack(
            HEADER_FMT,
            len(handle),
            N,
            frames.dtype.itemsize,
            NUM_SLOTS,
        )

        buf[:HEADER_SIZE] = header
        buf[HANDLE_OFFSET : HANDLE_OFFSET + len(handle)] = handle

        write_i64(buf, LATEST_OFFSET, -1)
        write_i64(buf, STOP_OFFSET, 0)

        for slot in range(NUM_SLOTS):
            write_i64(buf, gen_offset(slot), 0)

        print("[A] GPU ring buffer ptr:", hex(frames.data.ptr), flush=True)
        print("[A] CUDA IPC handle bytes:", len(handle), flush=True)

        ready.set()

        for seq in range(NUM_FRAMES):
            slot = seq % NUM_SLOTS

            # Mark slot as being written.
            # Odd generation means "writer active".
            write_i64(buf, gen_offset(slot), seq * 2 + 1)

            # GPU write.
            # Frame seq=7 contains [7000, 7001, ..., 7015]
            frames[slot] = base + seq * 1000

            # Make GPU write visible before publishing.
            cp.cuda.runtime.deviceSynchronize()

            # Mark slot stable.
            # Even generation means "safe to read".
            write_i64(buf, gen_offset(slot), seq * 2 + 2)

            # Publish latest frame.
            write_i64(buf, LATEST_OFFSET, seq)

            print(f"[A] published seq={seq:02d} slot={slot}", flush=True)
            time.sleep(PRODUCER_SLEEP_SEC)

        # Tell all consumers no more frames are coming.
        write_i64(buf, STOP_OFFSET, 1)
        print("[A] stop published; waiting for consumers to close", flush=True)

        # Important: producer must keep the original GPU allocation alive
        # until all consumers close their IPC mappings.
        for ev in consumer_done_events:
            ev.wait(timeout=30)

        print("[A] all consumers done; producer exiting", flush=True)

    finally:
        shm.close()


def consumer(shm_name, ready, done, consumer_id):
    import cupy as cp

    cp.cuda.Device(0).use()

    shm = SharedMemory(name=shm_name)
    buf = shm.buf

    ptr = None
    last_seen = -1
    accepted = 0
    dropped = 0

    try:
        ready.wait(timeout=30)

        handle_len, n, itemsize, num_slots = struct.unpack(
            HEADER_FMT,
            buf[:HEADER_SIZE],
        )

        handle = bytes(buf[HANDLE_OFFSET : HANDLE_OFFSET + handle_len])

        # Each consumer opens the same IPC handle independently.
        ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)

        nbytes = num_slots * n * itemsize

        # Wrap remote GPU memory.
        # Consumer only reads from this array by convention.
        owner = object()
        mem = cp.cuda.UnownedMemory(ptr, nbytes, owner)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        frames = cp.ndarray((num_slots, n), dtype=cp.float32, memptr=memptr)

        print(f"[C{consumer_id}] opened remote ptr: {hex(ptr)}", flush=True)

        while True:
            latest = read_i64(buf, LATEST_OFFSET)
            stop = read_i64(buf, STOP_OFFSET)

            if latest > last_seen:
                # Option A: process every frame you can still validate.
                # If producer already reused the slot, count it as dropped.
                for seq in range(last_seen + 1, latest + 1):
                    slot = seq % num_slots
                    expected_gen = seq * 2 + 2

                    g1 = read_i64(buf, gen_offset(slot))

                    if g1 != expected_gen or g1 % 2 == 1:
                        dropped += 1
                        last_seen = seq
                        continue

                    # Read-only GPU access.
                    # For real work, launch a kernel on frames[slot] instead
                    # of copying to CPU.
                    arr = cp.asnumpy(frames[slot])

                    g2 = read_i64(buf, gen_offset(slot))

                    if g1 == g2 == expected_gen:
                        print(
                            f"[C{consumer_id}] read seq={seq:02d} "
                            f"slot={slot} first={arr[0]:.0f} last={arr[-1]:.0f}",
                            flush=True,
                        )
                        accepted += 1
                    else:
                        dropped += 1

                    last_seen = seq

            if stop and latest <= last_seen:
                break

            time.sleep(CONSUMER_SLEEP_SEC)

        print(
            f"[C{consumer_id}] done. accepted={accepted}, dropped={dropped}",
            flush=True,
        )

        del frames, memptr, mem

    finally:
        if ptr is not None:
            cp.cuda.runtime.ipcCloseMemHandle(ptr)
        shm.close()
        done.set()


def main():
    set_start_method("spawn", force=True)

    shm = SharedMemory(create=True, size=SHM_SIZE)

    ready = Event()
    consumer_done_events = [Event() for _ in range(NUM_CONSUMERS)]

    try:
        p_a = Process(
            target=producer,
            args=(shm.name, ready, consumer_done_events),
        )

        consumers = [
            Process(
                target=consumer,
                args=(shm.name, ready, consumer_done_events[i], i),
            )
            for i in range(NUM_CONSUMERS)
        ]

        p_a.start()

        for p in consumers:
            p.start()

        p_a.join()

        for p in consumers:
            p.join()

        exitcodes = [p_a.exitcode] + [p.exitcode for p in consumers]
        if any(code != 0 for code in exitcodes):
            raise RuntimeError(f"process failed; exitcodes={exitcodes}")

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()