"""Observability: stats(), miss reasons, explain(), invalidate().

A cache that is hard to observe is hard to trust — every miss carries a
reason, and explain() answers "what would this call do, and why" without
executing anything.
"""

import time

from cachau import cache


@cache(ttl="2s")
def forecast(city: str) -> str:
    print(f"  fetching forecast for {city} ...")
    return f"{city}: sunny"


def show(title: str) -> None:
    stats = forecast.cache.stats()
    print(
        f"[{title}] hits={stats.hits} misses={stats.misses} "
        f"(not_found={stats.miss_not_found}, expired={stats.miss_expired}, "
        f"invalidated={stats.miss_invalidated})"
    )


def main() -> None:
    forecast("montevideo")  # miss_not_found
    forecast("montevideo")  # HIT
    show("after first calls")

    print("\nexplain before expiry:")
    print(forecast.cache.explain("montevideo"))

    print("\nwaiting for the 2s TTL to lapse ...")
    time.sleep(2.1)
    print("explain after expiry (pure observation - nothing is recomputed):")
    print(forecast.cache.explain("montevideo"))

    forecast("montevideo")  # miss_expired: recomputed, fresh TTL
    show("after expiry")

    forecast.cache.invalidate("montevideo")
    forecast("montevideo")  # miss_invalidated
    show("after invalidate")

    stats = forecast.cache.stats()
    print(f"\nentries={stats.entries} writes={stats.writes}")
    print("every number above is an immutable CacheStats snapshot: stats() never lies")


if __name__ == "__main__":
    main()
