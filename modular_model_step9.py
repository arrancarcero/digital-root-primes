import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

Row = Dict[str, float | int | None]


def is_prime(n: int) -> bool:
    """Return True if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(n**0.5) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def generate_primes(count: int) -> List[int]:
    """Generate the first `count` primes."""
    if count < 0:
        raise ValueError("count must be non-negative")

    primes: List[int] = []
    candidate = 2

    while len(primes) < count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1

    return primes


def digital_root(n: int) -> int:
    """Return the digital root of n."""
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def signed_mod9(n: int) -> int:
    """
    Signed mod-9 mapping:
    1 -> -8, 2 -> -7, ..., 8 -> -1, 0 -> 0
    """
    r = n % 9
    if r == 0:
        return 0
    return r - 9


def angle_mod9(n: int) -> float:
    """Map mod-9 residue to angle on unit circle."""
    r = n % 9
    return 2 * math.pi * r / 9


def residue_role_mod9(n: int) -> int:
    """
    Role labels:
    0 = prime-eligible residue
    1 = divisible-by-3 barrier
    2 = special prime 3
    """
    if n == 3:
        return 2
    r = n % 9
    if r in {0, 3, 6}:
        return 1
    return 0


def build_prime_dataframe(count: int) -> List[Row]:
    """Build a symbolic feature table for primes."""
    primes = generate_primes(count)
    rows: List[Row] = []

    for i, p in enumerate(primes):
        prev_gap = None
        next_gap = None
        next_signed_state = None

        if i > 0:
            prev_gap = p - primes[i - 1]

        if i < len(primes) - 1:
            next_gap = primes[i + 1] - p
            next_signed_state = signed_mod9(primes[i + 1])

        theta = angle_mod9(p)

        rows.append(
            {
                "prime": p,
                "mod3": p % 3,
                "mod9": p % 9,
                "digital_root": digital_root(p),
                "signed_mod9": signed_mod9(p),
                "fractional_mod9": signed_mod9(p) / 9,
                "angle": theta,
                "cos_angle": math.cos(theta),
                "sin_angle": math.sin(theta),
                "is_even": int(p % 2 == 0),
                "prev_gap": prev_gap,
                "next_gap": next_gap,
                "role_mod9": residue_role_mod9(p),
                "target_next_signed_mod9": next_signed_state,
            }
        )

    return rows


def clean_dataframe_for_learning(rows: Sequence[Row]) -> List[Row]:
    """
    Clean the feature table for sequence learning.
    We drop rows where needed values are missing.
    """
    cleaned_rows: List[Row] = []

    for row in rows:
        prime = int(row["prime"])
        if prime <= 3:
            continue

        required_values = (
            row["prev_gap"],
            row["next_gap"],
            row["target_next_signed_mod9"],
        )
        if any(value is None for value in required_values):
            continue

        cleaned = dict(row)
        cleaned["target_next_signed_mod9"] = int(cleaned["target_next_signed_mod9"])
        cleaned_rows.append(cleaned)

    return cleaned_rows


def build_feature_matrix(rows: Sequence[Row]) -> Tuple[List[List[float]], List[int]]:
    """Select model features and target."""
    feature_columns = [
        "prime",
        "mod3",
        "mod9",
        "digital_root",
        "signed_mod9",
        "fractional_mod9",
        "cos_angle",
        "sin_angle",
        "prev_gap",
        "next_gap",
        "role_mod9",
    ]

    X: List[List[float]] = []
    y: List[int] = []

    for row in rows:
        X.append([float(row[column]) for column in feature_columns])
        y.append(int(row["target_next_signed_mod9"]))

    return X, y


def build_sequence_windows(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    window_size: int = 5,
) -> Tuple[List[List[List[float]]], List[int]]:
    """
    Build sliding windows for sequence prediction.

    Each input sample is a window of `window_size` rows of features.
    The target is the next row's target value.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must be the same length")

    sequences: List[List[List[float]]] = []
    targets: List[int] = []

    for i in range(len(X) - window_size):
        window = [list(row) for row in X[i : i + window_size]]
        target = y[i + window_size]

        sequences.append(window)
        targets.append(target)

    return sequences, targets


def main() -> None:
    rows = build_prime_dataframe(1000)
    rows = clean_dataframe_for_learning(rows)

    print("Cleaned dataframe head:")
    for row in rows[:10]:
        print(row)
    print()

    X, y = build_feature_matrix(rows)

    print("Feature matrix shape:", (len(X), len(X[0]) if X else 0))
    print("Target shape:", (len(y),))
    print()

    window_size = 5
    sequences, targets = build_sequence_windows(X, y, window_size=window_size)

    print(f"Built {len(sequences)} sequences with window size {window_size}")
    print()

    print("Example sequence[0]:")
    if sequences:
        for step, row in enumerate(sequences[0]):
            print(f"Step {step}: {row}")

    print()
    if targets:
        print("Example target[0]:", targets[0])

    print()
    print("Unique target classes:")
    print(sorted(set(targets)))

    print("\nTarget frequency counts:")
    print(Counter(targets))


if __name__ == "__main__":
    main()
