"""Keying cost by argument type.

The fundamental rule (GUIDELINES.md §15):

    T_key + T_lookup + T_deserialize < T_recompute

T_key is the first term, and it scales with argument size — these numbers
tell you when hashing the argument costs more than your function does.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _timing import fmt, measure, print_environment

from cachau.keys import digest_arguments


def sample(a):
    return a


def main() -> None:
    print("Keying cost (median): digest_arguments over one argument\n")
    print_environment()

    cases = [("int", 42), ("str (10 chars)", "x" * 10), ("dict (10 items)", {i: i for i in range(10)})]

    try:
        import numpy as np

        cases += [
            ("ndarray 1 KB (128 f64)", np.arange(128, dtype=np.float64)),
            ("ndarray 800 KB (100k f64)", np.arange(100_000, dtype=np.float64)),
            ("ndarray 8 MB (1M f64)", np.arange(1_000_000, dtype=np.float64)),
            ("ndarray 80 MB (10M f64)", np.arange(10_000_000, dtype=np.float64)),
        ]
    except ImportError:
        pass

    try:
        import numpy as np
        import pandas as pd

        cases += [
            (
                "DataFrame 3 cols x 100k rows",
                pd.DataFrame(
                    {
                        "a": np.arange(100_000),
                        "b": np.random.default_rng(0).random(100_000),
                        "c": ["x"] * 100_000,
                    }
                ),
            ),
        ]
    except ImportError:
        pass

    for label, value in cases:
        cost = measure(lambda v=value: digest_arguments(sample, (v,), {}))
        print(f"  {label:<30} {fmt(cost)}")

    print(
        "\nreading: if your function is FASTER than its row above, caching "
        "loses.\nEscape hatch: key=lambda ...: <cheap stable identity>."
    )


if __name__ == "__main__":
    main()
