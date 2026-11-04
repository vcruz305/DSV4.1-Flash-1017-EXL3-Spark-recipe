#!/usr/bin/env python3
"""Page-cache residency of files (mincore), e.g. the cold-expert side file.

  python3 tools/pagecache_resident.py <file>...          # resident GiB / size
  python3 tools/pagecache_resident.py --willneed <file>  # also re-request read-ahead
"""
import ctypes, mmap, os, sys

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
PAGE = os.sysconf("SC_PAGE_SIZE")
CHUNK = 1 << 30


def resident(path):
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    res = 0
    try:
        for off in range(0, size, CHUNK):
            n = min(CHUNK, size - off)
            addr = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, off)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), "mmap")
            vec = (ctypes.c_ubyte * ((n + PAGE - 1) // PAGE))()
            try:
                if libc.mincore(addr, n, vec) != 0:
                    raise OSError(ctypes.get_errno(), "mincore")
                res += sum(b & 1 for b in bytes(vec)) * PAGE
            finally:
                libc.munmap(addr, n)
    finally:
        os.close(fd)
    return min(res, size), size


def main():
    args = sys.argv[1:]
    need = "--willneed" in args
    for p in [a for a in args if a != "--willneed"]:
        r, s = resident(p)
        print(f"{r / 2**30:7.2f} / {s / 2**30:7.2f} GiB resident  {p}")
        if need:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            os.close(fd)


if __name__ == "__main__":
    main()
