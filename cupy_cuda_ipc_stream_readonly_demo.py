import time
import struct
from multiprocessing import Process, Event, set_start_method
from multiprocessing.shared_memory import SharedMemory

# Streaming parameters
N = 16
NUM_SLOTS = 4
NUM_FRAMES = 20
PRODUCER_SLEEP_SEC = 0.05
CONSUMER_SLEEP_SEC = 0.02

# Shared-memory layout
HEADER_FMT = "IQQI"  # handle_len:uint32, n:uint64, itemsize:uint64, num_slots:uint32
HEADER_SIZE = struct.calcsize(HEADER_FMT)

HANDLE_OFFSET = 128
CTRL_OFFSET = 512

LATEST_OFFSET = CTRL_OFFSET          # int64, latest published seq, starts at -1
STOP_OFFSET = CTRL_OFFSET + 8        # int64, 0/1
GEN_OFFSET = CTRL_OFFSET + 16        # int64[NUM_SLOTS]

SHM_SIZE = 4096


def write_i64(buf, offset, value):
    struct.pack_into("q", buf, offset, value)


def read_i64(buf, offset):
    return struct.unpack_from("q", buf, offset)[0]


def gen_offset(slot):
    return GEN_OFFSET + slot * 8


def producer(shm_name, ready):
    import cupy as cp

    cp.cuda.Device(0).use()

    # One GPU allocation containing multiple frame slots.
    frames = cp.empty((NUM_SLOTS, N), dtype=cp.float32)
    base = cp.arange(N, dtype=cp.float32)

    # Export CUDA IPC handle for the base allocation.
    handle = cp.cuda.runtime.ipcGetMemHandle(frames.data.ptr)

    shm = SharedMemory(name=shm_name)
    buf = shm.buf

    try:
        # Write metadata + CUDA IPC handle into host shm.
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

        for s in range(NUM_SLOTS):
            write_i64(buf, gen_offset(s), 0)

        print("[A] GPU ring buffer ptr:", hex(frames.data.ptr), flush=True)
        print("[A] CUDA IPC handle bytes:", len(handle), flush=True)

        ready.set()

        for seq in range(NUM_FRAMES):
            slot = seq % NUM_SLOTS

            # Mark this slot as being written.
            # Odd generation means "writer active".
            write_i64(buf, gen_offset(slot), seq * 2 + 1)

            # GPU write: frame contents are easy to verify.
            # Frame seq=7 contains [7000, 7001, 7002, ...]
            frames[slot] = base + seq * 1000

            # Make sure GPU write is complete before publishing.
            cp.cuda.runtime.deviceSynchronize()

            # Mark slot as stable.
            # Even generation means "safe to read".
            write_i64(buf, gen_offset(slot), seq * 2 + 2)

            # Publish latest frame sequence.
            write_i64(buf, LATEST_OFFSET, seq)

            print(f"[A] published seq={seq:02d} slot={slot}", flush=True)

            time.sleep(PRODUCER_SLEEP_SEC)

        write_i64(buf, STOP_OFFSET, 1)

        # Keep frames alive briefly so consumer can finish reading final frames.
        time.sleep(1.0)

    finally:
        shm.close()


def consumer(shm_name, ready):
    import cupy as cp

    cp.cuda.Device(0).use()

    shm = SharedMemory(name=shm_name)
    buf = shm.buf

    ptr = None
    last_seen = -1
    accepted = 0
    rejected = 0

    try:
        ready.wait(timeout=30)

        handle_len, n, itemsize, num_slots = struct.unpack(
            HEADER_FMT,
            buf[:HEADER_SIZE],
        )

        handle = bytes(buf[HANDLE_OFFSET : HANDLE_OFFSET + handle_len])

        # Open remote GPU allocation in this process.
        ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)

        nbytes = num_slots * n * itemsize

        # Wrap imported pointer as a read-only-by-convention CuPy array.
        owner = object()
        mem = cp.cuda.UnownedMemory(ptr, nbytes, owner)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        frames = cp.ndarray((num_slots, n), dtype=cp.float32, memptr=memptr)

        print("[B] opened remote ptr:", hex(ptr), flush=True)

        while True:
            latest = read_i64(buf, LATEST_OFFSET)
            stop = read_i64(buf, STOP_OFFSET)

            if latest > last_seen:
                # Consume all unseen sequence numbers.
                # You can also change this to consume only `latest`
                # if you want "latest-frame-only" behavior.
                for seq in range(last_seen + 1, latest + 1):
                    slot = seq % num_slots

                    g1 = read_i64(buf, gen_offset(slot))

                    # If odd, producer is currently writing that slot.
                    if g1 % 2 == 1:
                        rejected += 1
                        continue

                    # This generation should correspond to this seq.
                    expected_gen = seq * 2 + 2
                    if g1 != expected_gen:
                        # Slot was already reused. Frame dropped.
                        rejected += 1
                        continue

                    # Read GPU data. Consumer does not write GPU memory.
                    arr = cp.asnumpy(frames[slot])

                    # Re-check generation after read.
                    # If changed, producer overwrote while we were reading.
                    g2 = read_i64(buf, gen_offset(slot))

                    if g1 == g2 == expected_gen:
                        print(
                            f"[B] read seq={seq:02d} slot={slot} "
                            f"first={arr[0]:.0f} last={arr[-1]:.0f}",
                            flush=True,
                        )
                        accepted += 1
                        last_seen = seq
                    else:
                        rejected += 1

            if stop and latest <= last_seen:
                break

            time.sleep(CONSUMER_SLEEP_SEC)

        print(
            f"[B] done. accepted={accepted}, rejected/dropped={rejected}",
            flush=True,
        )

        del frames, memptr, mem

    finally:
        if ptr is not None:
            cp.cuda.runtime.ipcCloseMemHandle(ptr)
        shm.close()


def main():
    set_start_method("spawn", force=True)

    shm = SharedMemory(create=True, size=SHM_SIZE)
    ready = Event()

    try:
        p_a = Process(target=producer, args=(shm.name, ready))
        p_b = Process(target=consumer, args=(shm.name, ready))

        p_a.start()
        p_b.start()

        p_a.join()
        p_b.join()

        if p_a.exitcode != 0 or p_b.exitcode != 0:
            raise RuntimeError(
                f"child failed: producer={p_a.exitcode}, consumer={p_b.exitcode}"
            )

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()