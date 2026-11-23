from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

GraphType = Literal[
    "adjacency_list",
    "adjacency_matrix",
    "incidence_matrix",
    "directed_incidence_matrix",
]

Weight = int | float


class Neighbor(NamedTuple):
    v: int
    w: Weight = 1


AdjacencyList = list[list[Neighbor]]
AdjacencyListInput = Sequence[Sequence[tuple[int, int] | tuple[int, float]]]
AdjacencyMatrix = NDArray[np.int64 | np.float64]
IncidenceMatrix = NDArray[np.int64 | np.float64]
MatrixInput = (
    NDArray[np.integer | np.floating]
    | Sequence[Sequence[int]]
    | Sequence[Sequence[float]]
)
