#!/usr/bin/env python3
"""Warm latency benchmark for POST /api/predict against a running server.
Usage: .venv/bin/python scripts/benchmark.py [base_url] [n]
Reports mean/p50/p95 ms with warm-up discarded. State hardware in docs."""
import json, statistics, sys, time, urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8101"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
payload = {"MedInc": 5.0, "HouseAge": 20, "AveRooms": 6.0, "AveBedrms": 1.0,
           "Population": 1000, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23}

def call():
    req = urllib.request.Request(base + "/api/predict",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        r.read()
    return (time.perf_counter() - t) * 1000

for _ in range(10):  # warm-up
    call()
times = [call() for _ in range(n)]
times.sort()
print(json.dumps({"n": n, "warmup": 10, "mean_ms": round(statistics.mean(times), 2),
                  "p50_ms": round(times[n // 2], 2), "p95_ms": round(times[int(n * 0.95) - 1], 2),
                  "concurrency": 1}, indent=2))
