from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def is_graphic_sequence(sequence: Sequence[int] | NDArray[np.integer]) -> bool:
    a: list[int] = []
    for value in sequence:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Sequence elements must be integers, got {value!r}")
        a.append(int(value))

    if sum(a) % 2:
        return False

    n = len(a)
    while a:
        a.sort(reverse=True)
        if a[-1] < 0 or a[0] >= n:
            return False
        k = a[0]
        if k == 0:
            return True
        a[0] = 0
        for i in range(1, k + 1):
            a[i] -= 1
    return True
