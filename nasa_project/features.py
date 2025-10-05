from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BasicFeatures:
    n: int
    mean: float
    std: float
    min: float
    max: float


def extract_basic(time, flux) -> BasicFeatures:
    # Accept masked arrays; treat masked as invalid
    f = flux
    if np.ma.isMaskedArray(f):
        f = f.filled(np.nan)
    f = np.asarray(f, dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return BasicFeatures(0, np.nan, np.nan, np.nan, np.nan)
    return BasicFeatures(
        n=int(f.size),
        mean=float(np.nanmean(f)),
        std=float(np.nanstd(f)),
        min=float(np.nanmin(f)),
        max=float(np.nanmax(f)),
    )
