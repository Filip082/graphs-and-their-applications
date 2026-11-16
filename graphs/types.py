from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

GraphType = Literal["adjacency_list", "adjacency_matrix", "incidence_matrix"]


class Neighbor(NamedTuple):
    v: int
    w: int = 1


AdjacencyList = list[list[Neighbor]]
AdjacencyListInput = Sequence[Sequence[tuple[int, int]]]
AdjacencyMatrix = NDArray[np.int64]
MatrixInput = NDArray[np.integer] | Sequence[Sequence[int]]


class IncidenceMatrix(NamedTuple):
    matrix: NDArray[np.int64]
    directed: bool = False


IncidenceMatrixInput = IncidenceMatrix | MatrixInput
