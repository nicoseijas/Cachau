"""Data workflows: DataFrame arguments, ignore= and key=.

Requires pandas (pip install pandas).
"""

import sys
import time

try:
    import pandas as pd
except ImportError:
    sys.exit("this example needs pandas: pip install pandas")

from cachau import cache

# --- 1. DataFrames are first-class cache keys -------------------------------
# Identity = columns + dtypes + content. Two equal frames share an entry even
# when they are different objects; changing one value is a different key.


@cache(max_memory="500MB")
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  building features ...")
    time.sleep(0.5)
    return df.assign(total=df["price"] * df["quantity"])


# --- 2. ignore=: collaborators that are not part of the result --------------


@cache(ignore=["logger"])
def run_pipeline(df: pd.DataFrame, logger=None) -> float:
    if logger:
        logger("running")
    return float(df["price"].sum())


# --- 3. key=: you know the identity better than any hasher ------------------
# Hashing a huge frame on every call can cost more than it saves. If a version
# tag fully determines the result, say so.


@cache(key=lambda df, version: version)
def train_model(df: pd.DataFrame, version: str) -> str:
    print(f"  training on version {version} ...")
    time.sleep(0.5)
    return f"model-{version}"


def main() -> None:
    sales = pd.DataFrame({"price": [10.0, 20.0], "quantity": [3, 1]})

    build_features(sales)
    build_features(sales.copy())  # equal content, different object: HIT
    print("features:", build_features.cache.stats().hit_rate == 0.5 and "1 compute, 1 hit")

    run_pipeline(sales, logger=print)
    run_pipeline(sales, logger=None)  # logger ignored: HIT
    print("pipeline:", run_pipeline.cache.stats().hits, "hit despite different logger")

    train_model(sales, "v1")
    train_model(sales.assign(price=[99.0, 99.0]), "v1")  # same version: HIT by design
    train_model(sales, "v2")
    print("model:", train_model.cache.stats().misses, "trainings for 3 calls")


if __name__ == "__main__":
    main()
