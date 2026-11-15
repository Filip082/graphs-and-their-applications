from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from graphs import convert
from graphs.convert import AdjacencyList, AdjacencyMatrix, IncidenceMatrix

GraphType = Literal["adjacency_list", "adjacency_matrix", "incidence_matrix"]


class Graph:
    vertex_count: int
    data_type: GraphType
    data: AdjacencyList | AdjacencyMatrix | IncidenceMatrix

    def __init__(
        self,
        data: Sequence[Sequence[int]] | NDArray[np.integer],
        type: GraphType,
    ) -> None:
        if type == "adjacency_list":
            if isinstance(data, np.ndarray):
                raise ValueError("Adjacency list must be a sequence, not an ndarray")
            normalized_list = convert._validate_adjacency_list(data)
            self.vertex_count = len(normalized_list)
            self.data = normalized_list
        elif type == "adjacency_matrix":
            normalized_matrix = convert._validate_adjacency_matrix(data)
            self.vertex_count = int(normalized_matrix.shape[0])
            self.data = normalized_matrix
        elif type == "incidence_matrix":
            normalized_inc = convert._validate_incidence_matrix(data)
            self.vertex_count = int(normalized_inc.shape[0])
            self.data = normalized_inc
        self.data_type = type

    @staticmethod
    def from_file(path: str | Path, type: GraphType) -> "Graph":
        text = Path(path).read_text(encoding="utf-8")
        if type == "adjacency_list":
            lines = text.splitlines()
            parsed_list: AdjacencyList = []
            for line in lines:
                content = line.split(":", 1)[1] if ":" in line else line
                parsed_list.append([int(tok) for tok in content.split()])
            return Graph(parsed_list, type)

        if type in ("adjacency_matrix", "incidence_matrix"):
            rows = [
                [int(tok) for tok in line.split()]
                for line in text.splitlines()
                if line.strip()
            ]
            if not rows:
                return Graph(np.zeros((0, 0), dtype=np.int64), type)
            row_lengths = {len(row) for row in rows}
            if len(row_lengths) != 1:
                raise ValueError("All matrix rows in file must have equal length")
            return Graph(np.array(rows, dtype=np.int64), type)

        raise ValueError(f"Unsupported graph data type: {type!r}")

    @staticmethod
    def create_random_binomial(n: int, p: float) -> "Graph":
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        if not 0.0 <= p <= 1.0:
            raise ValueError("Probability p must be between 0.0 and 1.0")

        matrix = np.zeros((n, n), dtype=np.int64)
        if n > 1:
            rng = np.random.default_rng()
            rows, cols = np.triu_indices(n, k=1)
            mask = rng.random(len(rows)) < p
            matrix[rows[mask], cols[mask]] = 1
            matrix[cols[mask], rows[mask]] = 1
        return Graph(matrix, "adjacency_matrix")

    @staticmethod
    def create_random_uniform(n: int, l: int) -> "Graph":  # noqa: E741
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        max_edges = n * (n - 1) // 2
        if l < 0 or l > max_edges:
            raise ValueError(f"Number of edges l must be between 0 and {max_edges}")

        matrix = np.zeros((n, n), dtype=np.int64)
        if l > 0:
            rng = np.random.default_rng()
            rows, cols = np.triu_indices(n, k=1)
            chosen = rng.choice(len(rows), size=l, replace=False)
            matrix[rows[chosen], cols[chosen]] = 1
            matrix[cols[chosen], rows[chosen]] = 1
        return Graph(matrix, "adjacency_matrix")

    def adjacency_list(self) -> AdjacencyList:
        if self.data_type == "adjacency_list":
            return [list(neighbors) for neighbors in cast(AdjacencyList, self.data)]
        if self.data_type == "adjacency_matrix":
            return convert.adjacency_matrix_to_adjacency_list(
                cast(AdjacencyMatrix, self.data)
            )
        return convert.incidence_matrix_to_adjacency_list(
            cast(IncidenceMatrix, self.data)
        )

    def adjacency_matrix(self) -> AdjacencyMatrix:
        if self.data_type == "adjacency_matrix":
            return cast(AdjacencyMatrix, self.data).copy()
        if self.data_type == "adjacency_list":
            return convert.adjacency_list_to_adjacency_matrix(
                cast(AdjacencyList, self.data)
            )
        return convert.incidence_matrix_to_adjacency_matrix(
            cast(IncidenceMatrix, self.data)
        )

    def incidence_matrix(self) -> IncidenceMatrix:
        if self.data_type == "incidence_matrix":
            return cast(IncidenceMatrix, self.data).copy()
        if self.data_type == "adjacency_list":
            return convert.adjacency_list_to_incidence_matrix(
                cast(AdjacencyList, self.data)
            )
        return convert.adjacency_matrix_to_incidence_matrix(
            cast(AdjacencyMatrix, self.data)
        )
