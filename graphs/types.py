from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

GraphType = Literal["adjacency_list", "adjacency_matrix", "incidence_matrix"]
AdjacencyList = list[list[int]]
AdjacencyMatrix = NDArray[np.int64]
IncidenceMatrix = NDArray[np.int64]
MatrixInput = NDArray[np.integer] | Sequence[Sequence[int]]
