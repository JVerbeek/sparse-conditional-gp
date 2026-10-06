"""Download the "nearby" snow water equivalent setup from the NRCS AWDB REST API.

Target : Shrine Pass, CO snow course (manual, a few readings per winter, record since 1942).
Sources: three SNOTEL snow pillows 6-17 km away (automated, daily, records since ~1979).

Writes data/nearby.npz with arrays
    target_t, target_y            decimal year and SWE [inches] of the snow course
    source{i}_t, source{i}_y      same for pillow i
    target_label, source_labels   station names
No account or API key is needed.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
OUT = Path(__file__).parent / "data" / "nearby.npz"

TARGET = ("06K09:CO:SNOW", "Shrine Pass course")
SOURCES = [
    ("415:CO:SNTL", "Copper Mountain pillow (6 km)"),
    ("842:CO:SNTL", "Vail Mountain pillow (17 km)"),
    ("485:CO:SNTL", "Fremont Pass pillow (17 km)"),
]


def fetch(triplet, duration, begin="1930-01-01", end="2026-09-01"):
    """Return a pandas Series of SWE [inches] indexed by date for one station.

    duration is DAILY for SNOTEL pillows and SEMIMONTHLY for snow courses.
    """
    url = (f"{API}?stationTriplets={triplet}&elements=WTEQ&duration={duration}"
           f"&beginDate={begin}&endDate={end}")
    with urllib.request.urlopen(url, timeout=120) as r:
        payload = json.load(r)
    df = pd.DataFrame(payload[0]["data"][0]["values"])
    date_col = "collectionDate" if "collectionDate" in df else "date"
    s = pd.Series(df["value"].astype(float).values, index=pd.to_datetime(df[date_col]))
    return s.sort_index().dropna()


def decimal_year(idx):
    idx = pd.DatetimeIndex(idx)
    start = pd.to_datetime(idx.year.astype(str) + "-01-01")
    ndays = np.where(idx.is_leap_year, 366.0, 365.0)
    return idx.year + (idx - start).days / ndays


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    tgt = fetch(TARGET[0], "SEMIMONTHLY")
    print(f"target  {TARGET[1]}: {len(tgt)} readings, {tgt.index.min().date()} .. {tgt.index.max().date()}")
    arrays = {"target_t": decimal_year(tgt.index), "target_y": tgt.values, "target_label": TARGET[1]}
    for i, (triplet, label) in enumerate(SOURCES):
        src = fetch(triplet, "DAILY")
        print(f"source {i} {label}: {len(src)} daily values, {src.index.min().date()} .. {src.index.max().date()}")
        arrays[f"source{i}_t"] = decimal_year(src.index)
        arrays[f"source{i}_y"] = src.values
    arrays["source_labels"] = np.array([label for _, label in SOURCES])
    np.savez(OUT, **arrays)
    print("saved", OUT)
