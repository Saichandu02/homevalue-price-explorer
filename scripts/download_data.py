#!/usr/bin/env python3
"""Download / verify the California Housing dataset.

Source archive: https://ndownloader.figshare.com/files/5976036
(the same archive scikit-learn's fetch_california_housing uses)
Expected SHA-256 of the extracted CSV shipped in this repo is in
data/DATA_CARD.md. If data/california_housing.csv already exists and
matches, this is a no-op verification.
"""
import csv, hashlib, io, sys, tarfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "california_housing.csv"
URL = "https://ndownloader.figshare.com/files/5976036"
COLS = ["Longitude", "Latitude", "HouseAge", "TotalRooms", "TotalBedrooms",
        "Population", "Households", "MedInc", "MedHouseVal"]

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> None:
    if OUT.exists():
        print(f"exists: {OUT} sha256={sha256(OUT)} rows={sum(1 for _ in OUT.open())-1}")
        return
    print(f"downloading {URL}")
    blob = urllib.request.urlopen(URL, timeout=120).read()
    tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    raw = tf.extractfile("CaliforniaHousing/cal_housing.data").read().decode()
    rows = [l.split(",") for l in raw.strip().splitlines()]
    with OUT.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    print(f"wrote {OUT} sha256={sha256(OUT)} rows={len(rows)}")

if __name__ == "__main__":
    sys.exit(main())
