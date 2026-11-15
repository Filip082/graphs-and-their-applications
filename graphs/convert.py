from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

AdjacencyList = list[list[int]]
AdjacencyMatrix = NDArray[np.int64]
IncidenceMatrix = NDArray[np.int64]
MatrixInput = NDArray[np.integer] | Sequence[Sequence[int]]


def _validate_adjacency_list(adjacency_list: Sequence[Sequence[int]]) -> AdjacencyList:
    n = len(adjacency_list)
    normalized: AdjacencyList = []
    neighbor_sets: list[set[int]] = []

    for u, neighbors in enumerate(adjacency_list):
        seen: set[int] = set()
        for v in neighbors:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise ValueError(f"Invalid vertex index: {v!r}")
            vi = int(v)
            if vi < 0 or vi >= n:
                raise ValueError(f"Vertex index out of range: {vi}")
            if vi == u:
                raise ValueError(f"Self-loop detected at vertex {u}")
            if vi in seen:
                raise ValueError(f"Duplicate neighbor {vi} for vertex {u}")
            seen.add(vi)
        neighbor_sets.append(seen)
        normalized.append(sorted(seen))

    for u, seen in enumerate(neighbor_sets):
        for v in seen:
            if u not in neighbor_sets[v]:
                raise ValueError(f"Asymmetric edge between {u} and {v}")

    return normalized


def _validate_adjacency_matrix(
    adjacency_matrix: MatrixInput,
) -> AdjacencyMatrix:
    arr = np.asarray(adjacency_matrix, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Adjacency matrix must be a square 2D array")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("Adjacency matrix entries must be 0 or 1")
    if not np.all(np.diag(arr) == 0):
        raise ValueError("Adjacency matrix must not have self-loops")
    if not np.array_equal(arr, arr.T):
        raise ValueError("Adjacency matrix must be symmetric")
    return arr.copy()


def _validate_incidence_matrix(
    incidence_matrix: MatrixInput,
) -> IncidenceMatrix:
    arr = np.asarray(incidence_matrix, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError("Incidence matrix must be a 2D array")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("Incidence matrix entries must be 0 or 1")
    if arr.shape[1] > 0:
        if not np.all(arr.sum(axis=0) == 2):
            raise ValueError("Each column of an incidence matrix must sum to 2")
        edges = {tuple(np.flatnonzero(arr[:, e])) for e in range(arr.shape[1])}
        if len(edges) != arr.shape[1]:
            raise ValueError("Incidence matrix contains duplicate edges")
    return arr.copy()


def adjacency_list_to_adjacency_matrix(
    adjacency_list: Sequence[Sequence[int]],
) -> AdjacencyMatrix:
    adj = _validate_adjacency_list(adjacency_list)
    n = len(adj)
    matrix = np.zeros((n, n), dtype=np.int64)
    for u, neighbors in enumerate(adj):
        for v in neighbors:
            matrix[u, v] = 1
    return matrix


def adjacency_list_to_incidence_matrix(
    adjacency_list: Sequence[Sequence[int]],
) -> IncidenceMatrix:
    adj = _validate_adjacency_list(adjacency_list)
    n = len(adj)
    edges = [(u, v) for u, neighbors in enumerate(adj) for v in neighbors if u < v]
    matrix = np.zeros((n, len(edges)), dtype=np.int64)
    for edge_idx, (u, v) in enumerate(edges):
        matrix[u, edge_idx] = 1
        matrix[v, edge_idx] = 1
    return matrix


def adjacency_matrix_to_adjacency_list(
    adjacency_matrix: MatrixInput,
) -> AdjacencyList:
    matrix = _validate_adjacency_matrix(adjacency_matrix)
    return [np.flatnonzero(row).astype(int).tolist() for row in matrix]


def incidence_matrix_to_adjacency_list(
    incidence_matrix: MatrixInput,
) -> AdjacencyList:
    matrix = _validate_incidence_matrix(incidence_matrix)
    n, m = matrix.shape
    adj: AdjacencyList = [[] for _ in range(n)]
    for e in range(m):
        u, v = np.flatnonzero(matrix[:, e])
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    for neighbors in adj:
        neighbors.sort()
    return adj


def adjacency_matrix_to_incidence_matrix(
    adjacency_matrix: MatrixInput,
) -> IncidenceMatrix:
    matrix = _validate_adjacency_matrix(adjacency_matrix)
    n = matrix.shape[0]
    rows, cols = np.nonzero(np.triu(matrix, k=1))
    m = len(rows)
    inc = np.zeros((n, m), dtype=np.int64)
    if m > 0:
        edge_idx = np.arange(m)
        inc[rows, edge_idx] = 1
        inc[cols, edge_idx] = 1
    return inc


def incidence_matrix_to_adjacency_matrix(
    incidence_matrix: MatrixInput,
) -> AdjacencyMatrix:
    matrix = _validate_incidence_matrix(incidence_matrix)
    adj_matrix = matrix @ matrix.T
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix
