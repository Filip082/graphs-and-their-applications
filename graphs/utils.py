from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def is_graphic_sequence(sequence: Sequence[int] | NDArray[np.integer]) -> bool:
    a: list[int] = []
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Sequence elements must be integers, got {value!r}")
        a.append(int(value))

    n = len(a)
    if any(x < 0 for x in a):
        return False
    if sum(1 for x in a if x % 2 == 1) % 2 == 1:
        return False

    a.sort(reverse=True)
    while True:
        if all(x == 0 for x in a):
            return True
        if a[0] >= n or any(x < 0 for x in a):
            return False
        k = a[0]
        for i in range(1, k + 1):
            a[i] -= 1
        a[0] = 0
        a.sort(reverse=True)
