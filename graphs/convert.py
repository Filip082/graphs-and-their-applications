import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from graphs.types import (
    AdjacencyList,
    AdjacencyListInput,
    AdjacencyMatrix,
    IncidenceMatrix,
    MatrixInput,
    Neighbor,
    Weight,
)


def _parse_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Invalid {label}: {value!r}")
    return int(value)


def _parse_weight(value: object, label: str) -> Weight:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if not math.isfinite(val):
            raise ValueError(f"Invalid {label}: {value!r}")
        return val
    raise ValueError(f"Invalid {label}: {value!r}")


def _to_numeric_array(matrix: MatrixInput) -> NDArray[np.int64 | np.float64]:
    arr = np.asarray(matrix)
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError("Matrix elements must be integers or floats")
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float64, copy=True)
        if not np.all(np.isfinite(out)):
            raise ValueError("Matrix elements must be finite")
        return out
    return arr.astype(np.int64, copy=True)


def _validate_adjacency_list(adjacency_list: AdjacencyListInput) -> AdjacencyList:
    n = len(adjacency_list)
    neighbor_weights: list[dict[int, Weight]] = []

    for u, neighbors in enumerate(adjacency_list, start=1):
        weights: dict[int, Weight] = {}
        for item in neighbors:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Invalid adjacency list entry: {item!r}")
            v = _parse_int(item[0], "vertex index")
            w = _parse_weight(item[1], "edge weight")
            if v < 1 or v > n:
                raise ValueError(f"Vertex index out of range: {v}")
            if v == u:
                raise ValueError(f"Self-loop detected at vertex {u}")
            if v in weights:
                raise ValueError(f"Duplicate neighbor {v} for vertex {u}")
            if w == 0:
                raise ValueError(f"Edge weight cannot be 0 (edge {u}-{v})")
            weights[v] = w
        neighbor_weights.append(weights)

    return [
        sorted(Neighbor(v, w) for v, w in weights.items())
        for weights in neighbor_weights
    ]


def _validate_adjacency_matrix(
    adjacency_matrix: MatrixInput,
) -> AdjacencyMatrix:
    arr = _to_numeric_array(adjacency_matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Adjacency matrix must be a square 2D array")
    if not np.all(np.diag(arr) == 0):
        raise ValueError("Adjacency matrix must not have self-loops")
    return arr


def _validate_incidence_matrix(
    incidence_matrix: MatrixInput,
    directed: bool = False,
) -> IncidenceMatrix:
    arr = _to_numeric_array(incidence_matrix)
    if arr.ndim != 2:
        raise ValueError("Incidence matrix must be a 2D array")
    if arr.shape[1] > 0:
        if not np.all(np.count_nonzero(arr, axis=0) == 2):
            raise ValueError(
                "Each column of an incidence matrix must have 2 non-zero entries"
            )
        edges: set[tuple[int, int]] = set()
        for e in range(arr.shape[1]):
            u, v = (int(idx) for idx in np.flatnonzero(arr[:, e]))
            if directed:
                first_val = arr[u, e]
                if first_val not in (-1, 1):
                    raise ValueError(
                        "First non-zero entry in a directed incidence matrix column "
                        "must be -1 or 1"
                    )
                edge = (u, v) if first_val == -1 else (v, u)
            else:
                if arr[u, e] != arr[v, e]:
                    raise ValueError(
                        "Non-zero entries in an incidence matrix column must be equal"
                    )
                edge = (u, v)
            if edge in edges:
                raise ValueError("Incidence matrix contains duplicate edges")
            edges.add(edge)
    return arr


def _matrix_to_list(matrix: AdjacencyMatrix) -> AdjacencyList:
    is_float = np.issubdtype(matrix.dtype, np.floating)
    return [
        [
            Neighbor(int(v) + 1, float(row[v]) if is_float else int(row[v]))
            for v in np.flatnonzero(row)
        ]
        for row in matrix
    ]


def _matrix_to_incidence(matrix: AdjacencyMatrix) -> IncidenceMatrix:
    n = matrix.shape[0]
    dtype: Any = np.float64 if np.issubdtype(matrix.dtype, np.floating) else np.int64
    if np.array_equal(matrix, matrix.T):
        rows, cols = np.nonzero(np.triu(matrix, k=1))
        m = len(rows)
        inc = np.zeros((n, m), dtype=dtype)
        if m > 0:
            edge_idx = np.arange(m)
            inc[rows, edge_idx] = inc[cols, edge_idx] = matrix[rows, cols]
        return inc

    rows, cols = np.nonzero(matrix)
    m = len(rows)
    inc = np.zeros((n, m), dtype=dtype)
    is_float = np.issubdtype(matrix.dtype, np.floating)
    for edge_idx, (u, v) in enumerate(zip(rows, cols, strict=True)):
        w: Weight = float(matrix[u, v]) if is_float else int(matrix[u, v])
        if u < v:
            inc[u, edge_idx] = -1
            inc[v, edge_idx] = w
        else:
            inc[v, edge_idx] = 1
            inc[u, edge_idx] = -w
    return inc


def adjacency_list_to_adjacency_matrix(
    adjacency_list: AdjacencyListInput,
) -> AdjacencyMatrix:
    adj = _validate_adjacency_list(adjacency_list)
    n = len(adj)
    has_float = any(isinstance(nb.w, float) for nbs in adj for nb in nbs)
    dtype: Any = np.float64 if has_float else np.int64
    matrix = np.zeros((n, n), dtype=dtype)
    for u, neighbors in enumerate(adj):
        for nb in neighbors:
            matrix[u, nb.v - 1] = nb.w
    return matrix


def adjacency_list_to_incidence_matrix(
    adjacency_list: AdjacencyListInput,
) -> IncidenceMatrix:
    return _matrix_to_incidence(adjacency_list_to_adjacency_matrix(adjacency_list))


def adjacency_matrix_to_adjacency_list(
    adjacency_matrix: MatrixInput,
) -> AdjacencyList:
    return _matrix_to_list(_validate_adjacency_matrix(adjacency_matrix))


def incidence_matrix_to_adjacency_list(
    incidence_matrix: MatrixInput,
    directed: bool = False,
) -> AdjacencyList:
    return _matrix_to_list(
        incidence_matrix_to_adjacency_matrix(incidence_matrix, directed=directed)
    )


def adjacency_matrix_to_incidence_matrix(
    adjacency_matrix: MatrixInput,
) -> IncidenceMatrix:
    return _matrix_to_incidence(_validate_adjacency_matrix(adjacency_matrix))


def incidence_matrix_to_adjacency_matrix(
    incidence_matrix: MatrixInput,
    directed: bool = False,
) -> AdjacencyMatrix:
    inc = _validate_incidence_matrix(incidence_matrix, directed=directed)
    n, m = inc.shape
    dtype: Any = np.float64 if np.issubdtype(inc.dtype, np.floating) else np.int64
    adj = np.zeros((n, n), dtype=dtype)
    for e in range(m):
        u, v = (int(idx) for idx in np.flatnonzero(inc[:, e]))
        if directed:
            if inc[u, e] == -1:
                adj[u, v] = inc[v, e]
            else:
                adj[v, u] = -inc[v, e]
        else:
            adj[u, v] = adj[v, u] = inc[u, e]
    return adj
