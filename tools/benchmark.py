"""Throughput / memory benchmark.   python tools/benchmark.py [n_none] [n_spacy]"""
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import politeness_py as P  # noqa: E402

n_none = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
n_spacy = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
base = list(P.datasets.receptive_train().text)
texts = (base * (max(n_none, n_spacy) // len(base) + 1))
words = sum(len(t.split()) for t in texts[:n_none])
t = time.time(); df = P.politeness(texts[:n_none], parser="none"); a = time.time() - t
print(f"parser=none : {n_none} texts ({words / n_none:.0f} words avg) in {a:.1f}s = {n_none / a:.0f} texts/s")
P.spacy_initialize()
t = time.time(); df = P.politeness(texts[:n_spacy], parser="spacy"); b = time.time() - t
print(f"parser=spacy: {n_spacy} texts in {b:.1f}s = {n_spacy / b:.0f} texts/s; output {df.shape}")
print(f"peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
