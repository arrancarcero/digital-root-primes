"""
Can a sliding window over digital roots predict the digital root of the next prime?

Run:  python experiment.py

For each range (primes below 10^5, 10^6, 10^7) this trains on the first 80% of
primes and tests on the last 20%, then repeats the whole thing on a CONTROL
sequence of fake "primes" (random numbers that merely avoid multiples of 2, 3, 5, 7
and thin out at the same rate real primes do). If a model scores the same on the
fake sequence as on the real one, it has learned small-number sieving and gap
sizes, not anything specific to primes.

Models are exact count tables (every context seen in training -> how often each
next root followed it). For inputs this small that is the best any model can do
with the same information, so a neural net fed the same window cannot beat it.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOTS = np.array([1, 2, 4, 5, 7, 8])          # digital roots a prime > 3 can have
ROOT_TO_CLASS = np.full(9, -1)
ROOT_TO_CLASS[ROOTS] = np.arange(6)
GAP_CAP = 15                                   # gaps are even; gap/2 capped at 15 (gap >= 30)
TRAIN_FRACTION = 0.8


def sieve(limit: int) -> np.ndarray:
    """All primes below `limit`."""
    flags = np.ones(limit, dtype=bool)
    flags[:2] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if flags[i]:
            flags[i * i :: i] = False
    return np.flatnonzero(flags)


def fake_primes(limit: int, rng: np.random.Generator) -> np.ndarray:
    """Control: random integers coprime to 210, kept with the density of real primes."""
    n = np.arange(11, limit)
    n = n[(n % 2 != 0) & (n % 3 != 0) & (n % 5 != 0) & (n % 7 != 0)]
    keep_probability = (210 / 48) / np.log(n)    # 48 of every 210 integers are coprime to 210
    return n[rng.random(n.size) < keep_probability]


def digital_root(n: np.ndarray) -> np.ndarray:
    return 1 + (n - 1) % 9


def build_contexts(p: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, int]], np.ndarray]:
    """
    For every position i, build what each model is allowed to see (as one integer
    id per position) and the answer: the digital-root class of the NEXT number.
    """
    p = p[p > 10]
    cls = ROOT_TO_CLASS[p % 9]
    half_gap = np.minimum(np.diff(p) // 2, GAP_CAP)        # half_gap[j] = gap after p[j]
    start, stop = 6, p.size - 1                            # need 6 back, 1 forward
    idx = np.arange(start, stop)
    target = cls[idx + 1]

    contexts: dict[str, tuple[np.ndarray, int]] = {
        "always guess the most common root": (np.zeros(idx.size, dtype=np.int64), 1),
    }
    for k in range(1, 7):                                  # window of the last k roots
        ctx = np.zeros(idx.size, dtype=np.int64)
        for back in range(k):
            ctx = ctx * 6 + cls[idx - back]
        contexts[f"window of last {k} root{'s' if k > 1 else ''}"] = (ctx, 6**k)
    for k in range(1, 4):                                  # current root + last k gaps
        ctx = cls[idx].astype(np.int64)
        for back in range(1, k + 1):
            ctx = ctx * (GAP_CAP + 1) + half_gap[idx - back]
        contexts[f"current root + last {k} gap{'s' if k > 1 else ''}"] = (ctx, 6 * (GAP_CAP + 1) ** k)
    contexts["current root + position on mod-210 wheel"] = (cls[idx] * 210 + p[idx] % 210, 6 * 210)
    return contexts, target


def score(ctx: np.ndarray, n_ctx: int, target: np.ndarray) -> tuple[float, float]:
    """Train a count table on the first 80%, return (accuracy %, bits of surprise) on the rest."""
    split = int(target.size * TRAIN_FRACTION)
    counts = np.full((n_ctx, 6), 0.5)                      # light smoothing for unseen contexts
    np.add.at(counts, (ctx[:split], target[:split]), 1)
    prob = counts / counts.sum(axis=1, keepdims=True)
    test_ctx, test_y = ctx[split:], target[split:]
    accuracy = float((prob[test_ctx].argmax(axis=1) == test_y).mean() * 100)
    bits = float(-np.log2(prob[test_ctx, test_y]).mean())
    return accuracy, bits


def check_saved_data(path: Path) -> str:
    """Confirm the saved CSV (primes below 100,000 with digital roots) matches a fresh sieve."""
    if not path.exists():
        return f"{path} not found, skipped"
    with path.open() as f:
        rows = [(int(r["Prime"]), int(r["DigitalRoot"])) for r in csv.DictReader(f)]
    saved = np.array(rows)
    fresh = sieve(100_000)
    ok = saved.shape[0] == fresh.size and np.array_equal(saved[:, 0], fresh) \
        and np.array_equal(saved[:, 1], digital_root(fresh))
    return f"{path.name}: {saved.shape[0]:,} rows, {'matches' if ok else 'DOES NOT MATCH'} a fresh sieve"


def main() -> None:
    print(check_saved_data(Path(__file__).parent / "data/digital_root_prime_data/prime_digital_roots.csv"))
    print(f"Blind guessing among 6 roots: {100/6:.1f}% accuracy, {math.log2(6):.3f} bits\n")
    rng = np.random.default_rng(0)
    for limit in (10**5, 10**6, 10**7):
        real = sieve(limit)
        fake = fake_primes(limit, rng)
        real_ctx, real_y = build_contexts(real)
        fake_ctx, fake_y = build_contexts(fake)
        print(f"=== below {limit:,}: {real.size:,} primes (control: {fake.size:,} fake primes) ===")
        print(f"{'model':44s} {'real primes':>18s} {'fake primes':>18s}")
        for name in real_ctx:
            ra, rb = score(*real_ctx[name], real_y)
            fa, fb = score(*fake_ctx[name], fake_y)
            print(f"{name:44s} {ra:6.1f}% {rb:6.3f} bits {fa:6.1f}% {fb:6.3f} bits")
        print()


if __name__ == "__main__":
    main()
