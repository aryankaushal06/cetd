from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import OnsetParams


def detect_onset(
    dates, rain, params: OnsetParams
) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)
    wet = r >= params.wet_day_mm
    year = int(d.iloc[0].year)
    t0 = pd.Timestamp(year=year, month=params.start_month, day=params.start_day)
    t1 = pd.Timestamp(year=year, month=params.end_month, day=params.end_day)
    si = int(np.searchsorted(d.values, np.datetime64(t0)))
    ei = int(np.searchsorted(d.values, np.datetime64(t1), side="right"))
    last = len(r) - params.accum_days - params.lookahead_days
    stop = min(last, ei)
    if stop <= si:
        return None, None
    for t in range(si, stop):
        win = r[t : t + params.accum_days]
        if np.nansum(win) >= params.accum_mm and np.all(
            win >= params.wet_day_mm
        ):
            fut = wet[
                t
                + params.accum_days : t
                + params.accum_days
                + params.lookahead_days
            ]
            dry_run, ok = 0, True
            for w in fut:
                if not w:
                    dry_run += 1
                    if dry_run >= params.dry_spell_days:
                        ok = False
                        break
                else:
                    dry_run = 0
            if ok:
                return d.iloc[t], t
    return None, None


def detect_wet_spell(
    dates, rain, params: OnsetParams
) -> Tuple[Optional[pd.Timestamp], Optional[int], int]:
    """Return (date, index, search_start_idx). search_start_idx is needed to floor the walkback."""
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)
    year = int(d.iloc[0].year)
    t0 = pd.Timestamp(year=year, month=params.start_month, day=params.start_day)
    t1 = pd.Timestamp(year=year, month=params.end_month, day=params.end_day)
    si = int(np.searchsorted(d.values, np.datetime64(t0)))
    ei = int(np.searchsorted(d.values, np.datetime64(t1), side="right"))
    stop = min(len(r) - params.accum_days + 1, ei)
    for t in range(si, max(si, stop)):
        win = r[t : t + params.accum_days]
        if np.nansum(win) >= params.accum_mm and not np.any(
            win < params.wet_day_mm
        ):
            return d.iloc[t], t, si
    return None, None, si


def wet_spell_start(
    dates,
    rain,
    idx: Optional[int],
    wet_day_mm: float,
    search_start_idx: int = 0,
) -> Optional[pd.Timestamp]:
    """Walk back from idx through consecutive wet days to find the true spell start."""
    if idx is None:
        return None
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)
    if idx < 0 or idx >= len(r):
        return None
    if r[idx] < wet_day_mm:
        return d.iloc[idx]
    j = idx
    while j - 1 >= search_start_idx and r[j - 1] >= wet_day_mm:
        j -= 1
    return d.iloc[j]
