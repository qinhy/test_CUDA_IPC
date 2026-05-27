import struct
from multiprocessing import Process, Event, set_start_method
from multiprocessing.shared_memory import SharedMemory

N = 8
HEADER_FMT = "IQQ"  # handle_len:uint32, n:uint64, itemsize:uint64
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SHM_SIZE = 4096


def producer(shm_name, ready, done):
    import cupy as cp

    cp.cuda.Device(0).use()

    # Fresh contiguous allocation. Do not use a view/offset pointer here.
    x = cp.arange(N, dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

    handle = cp.cuda.runtime.ipcGetMemHandle(x.data.ptr)

    shm = SharedMemory(name=shm_name)
    try:
        header = struct.pack(HEADER_FMT, len(handle), N, x.dtype.itemsize)
        shm.buf[:HEADER_SIZE] = header
        shm.buf[HEADER_SIZE : HEADER_SIZE + len(handle)] = handle

        print("[A] before:", cp.asnumpy(x), flush=True)
        print("[A] ptr   :", hex(x.data.ptr), flush=True)
        print("[A] CUDA IPC handle bytes:", len(handle), flush=True)

        # Tell B the handle is ready.
        ready.set()

        # Keep x alive until B is done.
        if not done.wait(timeout=30):
            raise TimeoutError("consumer did not finish")

        cp.cuda.runtime.deviceSynchronize()
        print("[A] after :", cp.asnumpy(x), flush=True)

    finally:
        shm.close()


def consumer(shm_name, ready, done):
    import cupy as cp

    cp.cuda.Device(0).use()

    shm = SharedMemory(name=shm_name)
    ptr = None

    try:
        ready.wait(timeout=30)

        handle_len, n, itemsize = struct.unpack(
            HEADER_FMT, shm.buf[:HEADER_SIZE]
        )
        handle = bytes(shm.buf[HEADER_SIZE : HEADER_SIZE + handle_len])

        # Open remote GPU allocation in this process.
        ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)

        nbytes = n * itemsize

        # Wrap imported raw device pointer as a CuPy array.
        owner = object()
        mem = cp.cuda.UnownedMemory(ptr, nbytes, owner)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        y = cp.ndarray((n,), dtype=cp.float32, memptr=memptr)

        print("[B] sees  :", cp.asnumpy(y), flush=True)
        print("[B] ptr   :", hex(ptr), flush=True)

        # In-place write from process B.
        y += 1000
        cp.cuda.runtime.deviceSynchronize()

        print("[B] wrote :", cp.asnumpy(y), flush=True)

        # Drop array wrappers before closing IPC mapping.
        del y, memptr, mem

    finally:
        if ptr is not None:
            cp.cuda.runtime.ipcCloseMemHandle(ptr)
        shm.close()
        done.set()


def main():
    set_start_method("spawn", force=True)

    shm = SharedMemory(create=True, size=SHM_SIZE)
    ready = Event()
    done = Event()

    try:
        p_a = Process(target=producer, args=(shm.name, ready, done))
        p_b = Process(target=consumer, args=(shm.name, ready, done))

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