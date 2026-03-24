import math
from collections import Counter
from typing import List, Tuple

import pandas as pd


def is_prime(n: int) -> bool:
    """Return True if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    limit = int(n ** 0.5) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def generate_primes(count: int) -> List[int]:
    """Generate the first `count` primes."""
    primes = []
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


def build_prime_dataframe(count: int) -> pd.DataFrame:
    """
    Build a symbolic feature table for primes.
    """
    primes = generate_primes(count)
    rows = []

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

    df = pd.DataFrame(rows)
    return df


def clean_dataframe_for_learning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the feature table for sequence learning.
    We drop rows where needed values are missing.
    """
    df = df.copy()

    # For the main prime pattern study, skip 2 and 3
    df = df[df["prime"] > 3].copy()

    # Drop rows missing gaps or next target
    df = df.dropna(subset=["prev_gap", "next_gap", "target_next_signed_mod9"]).copy()

    # Convert targets back to integer
    df["target_next_signed_mod9"] = df["target_next_signed_mod9"].astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select model features and target.
    """
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

    X = df[feature_columns].copy()
    y = df["target_next_signed_mod9"].copy()
    return X, y


def build_sequence_windows(
    X: pd.DataFrame,
    y: pd.Series,
    window_size: int = 5,
) -> Tuple[List[List[List[float]]], List[int]]:
    """
    Build sliding windows for sequence prediction.

    Each input sample is a window of `window_size` rows of features.
    The target is the next row's target value.
    """
    X_values = X.values.tolist()
    y_values = y.tolist()

    sequences = []
    targets = []

    for i in range(len(X_values) - window_size):
        window = X_values[i : i + window_size]
        target = y_values[i + window_size]

        sequences.append(window)
        targets.append(target)

    return sequences, targets


def main() -> None:
    df = build_prime_dataframe(1000)
    df = clean_dataframe_for_learning(df)

    print("Cleaned dataframe head:")
    print(df.head(10))
    print()

    X, y = build_feature_matrix(df)

    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print()

    window_size = 5
    sequences, targets = build_sequence_windows(X, y, window_size=window_size)

    print(f"Built {len(sequences)} sequences with window size {window_size}")
    print()

    print("Example sequence[0]:")
    for step, row in enumerate(sequences[0]):
        print(f"Step {step}: {row}")

    print()
    print("Example target[0]:", targets[0])

    print()
    print("Unique target classes:")
    print(sorted(set(targets)))

    print("\nTarget frequency counts:")
    print(Counter(targets))


if __name__ == "__main__":
    main()
