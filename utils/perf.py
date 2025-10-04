from __future__ import annotations
import hashlib
import time
from contextlib import contextmanager
import numpy as np


def array_digest(a: np.ndarray) -> str:
    x = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(str(x.dtype).encode())
    h.update(x.view(np.uint8).data)
    return h.hexdigest()


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"[timer] {label}: {dt:.1f} ms")