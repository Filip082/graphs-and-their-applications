import numpy as np

from graphs.types import (
    AdjacencyList,
    AdjacencyListInput,
    AdjacencyMatrix,
    IncidenceMatrix,
    MatrixInput,
    Neighbor,
)


def _validate_adjacency_list(adjacency_list: AdjacencyListInput) -> AdjacencyList:
    n = len(adjacency_list)
    normalized: AdjacencyList = []
    neighbor_weights: list[dict[int, int]] = []

    for idx, neighbors in enumerate(adjacency_list):
        u = idx + 1
        weights: dict[int, int] = {}
        for item in neighbors:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Invalid adjacency list entry: {item!r}")
            v, w = item
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise ValueError(f"Invalid vertex index: {v!r}")
            vi = int(v)
            if vi < 1 or vi > n:
                raise ValueError(f"Vertex index out of range: {vi}")
            if vi == u:
                raise ValueError(f"Self-loop detected at vertex {u}")
            if vi in weights:
                raise ValueError(f"Duplicate neighbor {vi} for vertex {u}")
            if isinstance(w, bool) or not isinstance(w, (int, np.integer)):
                raise ValueError(f"Invalid edge weight: {w!r}")
            wi = int(w)
            if wi == 0:
                raise ValueError(f"Edge weight cannot be 0 (edge {u}-{vi})")
            weights[vi] = wi
        neighbor_weights.append(weights)
        normalized.append(sorted(Neighbor(v, w) for v, w in weights.items()))

    for idx, weights in enumerate(neighbor_weights):
        u = idx + 1
        for v, w in weights.items():
            if u not in neighbor_weights[v - 1]:
                raise ValueError(f"Asymmetric edge between {u} and {v}")
            if neighbor_weights[v - 1][u] != w:
                raise ValueError(f"Asymmetric edge weight between {u} and {v}")

    return normalized


def _validate_adjacency_matrix(
    adjacency_matrix: MatrixInput,
) -> AdjacencyMatrix:
    arr = np.asarray(adjacency_matrix, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Adjacency matrix must be a square 2D array")
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
    if arr.shape[1] > 0:
        if not np.all(np.count_nonzero(arr, axis=0) == 2):
            raise ValueError(
                "Each column of an incidence matrix must have 2 non-zero entries"
            )
        edges: set[tuple[int, ...]] = set()
        for e in range(arr.shape[1]):
            u, v = np.flatnonzero(arr[:, e])
            if arr[u, e] != arr[v, e]:
                raise ValueError(
                    "Non-zero entries in an incidence matrix column must be equal"
                )
            edges.add((int(u), int(v)))
        if len(edges) != arr.shape[1]:
            raise ValueError("Incidence matrix contains duplicate edges")
    return arr.copy()


def adjacency_list_to_adjacency_matrix(
    adjacency_list: AdjacencyListInput,
) -> AdjacencyMatrix:
    adj = _validate_adjacency_list(adjacency_list)
    n = len(adj)
    matrix = np.zeros((n, n), dtype=np.int64)
    for u_idx, neighbors in enumerate(adj):
        for neighbor in neighbors:
            matrix[u_idx, neighbor.v - 1] = neighbor.w
    return matrix


def adjacency_list_to_incidence_matrix(
    adjacency_list: AdjacencyListInput,
) -> IncidenceMatrix:
    adj = _validate_adjacency_list(adjacency_list)
    n = len(adj)
    edges = [
        (u_idx, neighbor.v - 1, neighbor.w)
        for u_idx, neighbors in enumerate(adj)
        for neighbor in neighbors
        if u_idx < neighbor.v - 1
    ]
    matrix = np.zeros((n, len(edges)), dtype=np.int64)
    for edge_idx, (u_idx, v_idx, weight) in enumerate(edges):
        matrix[u_idx, edge_idx] = weight
        matrix[v_idx, edge_idx] = weight
    return matrix


def adjacency_matrix_to_adjacency_list(
    adjacency_matrix: MatrixInput,
) -> AdjacencyList:
    matrix = _validate_adjacency_matrix(adjacency_matrix)
    return [
        [Neighbor(int(v_idx) + 1, int(row[v_idx])) for v_idx in np.flatnonzero(row)]
        for row in matrix
    ]


def incidence_matrix_to_adjacency_list(
    incidence_matrix: MatrixInput,
) -> AdjacencyList:
    matrix = _validate_incidence_matrix(incidence_matrix)
    n, m = matrix.shape
    adj: AdjacencyList = [[] for _ in range(n)]
    for e in range(m):
        u, v = np.flatnonzero(matrix[:, e])
        w = int(matrix[u, e])
        adj[int(u)].append(Neighbor(int(v) + 1, w))
        adj[int(v)].append(Neighbor(int(u) + 1, w))
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
        weights = matrix[rows, cols]
        inc[rows, edge_idx] = weights
        inc[cols, edge_idx] = weights
    return inc


def incidence_matrix_to_adjacency_matrix(
    incidence_matrix: MatrixInput,
) -> AdjacencyMatrix:
    matrix = _validate_incidence_matrix(incidence_matrix)
    n, m = matrix.shape
    adj_matrix = np.zeros((n, n), dtype=np.int64)
    for e in range(m):
        u, v = np.flatnonzero(matrix[:, e])
        w = matrix[u, e]
        adj_matrix[u, v] = w
        adj_matrix[v, u] = w
    return adj_matrix
