from __future__ import annotations

import random


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    random_state = random.Random(seed)
    bootstrapped = []
    for _ in range(samples):
        draw = [values[random_state.randrange(len(values))] for _ in range(len(values))]
        bootstrapped.append(sum(draw) / len(draw))
    bootstrapped.sort()
    lower_index = int((alpha / 2) * len(bootstrapped))
    upper_index = int((1 - alpha / 2) * len(bootstrapped)) - 1
    return (bootstrapped[lower_index], bootstrapped[upper_index])
