import numpy as np

from graphs.types import (
    AdjacencyList,
    AdjacencyListInput,
    AdjacencyMatrix,
    IncidenceMatrix,
    IncidenceMatrixInput,
    MatrixInput,
    Neighbor,
)


def _parse_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Invalid {label}: {value!r}")
    return int(value)


def _validate_adjacency_list(adjacency_list: AdjacencyListInput) -> AdjacencyList:
    n = len(adjacency_list)
    neighbor_weights: list[dict[int, int]] = []

    for u, neighbors in enumerate(adjacency_list, start=1):
        weights: dict[int, int] = {}
        for item in neighbors:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Invalid adjacency list entry: {item!r}")
            v = _parse_int(item[0], "vertex index")
            w = _parse_int(item[1], "edge weight")
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
    arr = np.asarray(adjacency_matrix, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Adjacency matrix must be a square 2D array")
    if not np.all(np.diag(arr) == 0):
        raise ValueError("Adjacency matrix must not have self-loops")
    return arr.copy()


def _coerce_incidence_input(
    incidence_matrix: IncidenceMatrixInput,
) -> tuple[np.ndarray, bool]:
    if isinstance(incidence_matrix, IncidenceMatrix):
        return np.asarray(incidence_matrix.matrix, dtype=np.int64), bool(
            incidence_matrix.directed
        )
    return np.asarray(incidence_matrix, dtype=np.int64), False


def _validate_incidence_matrix(
    incidence_matrix: IncidenceMatrixInput,
) -> IncidenceMatrix:
    arr, directed = _coerce_incidence_input(incidence_matrix)
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
                first_val = int(arr[u, e])
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
    return IncidenceMatrix(arr.copy(), directed=directed)


def _matrix_to_list(matrix: AdjacencyMatrix) -> AdjacencyList:
    return [
        [Neighbor(int(v) + 1, int(row[v])) for v in np.flatnonzero(row)]
        for row in matrix
    ]


def _matrix_to_incidence(matrix: AdjacencyMatrix) -> IncidenceMatrix:
    n = matrix.shape[0]
    if np.array_equal(matrix, matrix.T):
        rows, cols = np.nonzero(np.triu(matrix, k=1))
        m = len(rows)
        inc = np.zeros((n, m), dtype=np.int64)
        if m > 0:
            edge_idx = np.arange(m)
            inc[rows, edge_idx] = inc[cols, edge_idx] = matrix[rows, cols]
        return IncidenceMatrix(inc, directed=False)

    rows, cols = np.nonzero(matrix)
    m = len(rows)
    inc = np.zeros((n, m), dtype=np.int64)
    for edge_idx, (u, v) in enumerate(zip(rows, cols, strict=True)):
        w = int(matrix[u, v])
        if u < v:
            inc[u, edge_idx] = -1
            inc[v, edge_idx] = w
        else:
            inc[v, edge_idx] = 1
            inc[u, edge_idx] = -w
    return IncidenceMatrix(inc, directed=True)


def adjacency_list_to_adjacency_matrix(
    adjacency_list: AdjacencyListInput,
) -> AdjacencyMatrix:
    adj = _validate_adjacency_list(adjacency_list)
    n = len(adj)
    matrix = np.zeros((n, n), dtype=np.int64)
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
    incidence_matrix: IncidenceMatrixInput,
) -> AdjacencyList:
    return _matrix_to_list(incidence_matrix_to_adjacency_matrix(incidence_matrix))


def adjacency_matrix_to_incidence_matrix(
    adjacency_matrix: MatrixInput,
) -> IncidenceMatrix:
    return _matrix_to_incidence(_validate_adjacency_matrix(adjacency_matrix))


def incidence_matrix_to_adjacency_matrix(
    incidence_matrix: IncidenceMatrixInput,
) -> AdjacencyMatrix:
    inc = _validate_incidence_matrix(incidence_matrix)
    n, m = inc.matrix.shape
    adj = np.zeros((n, n), dtype=np.int64)
    for e in range(m):
        u, v = (int(idx) for idx in np.flatnonzero(inc.matrix[:, e]))
        if inc.directed:
            if inc.matrix[u, e] == -1:
                adj[u, v] = inc.matrix[v, e]
            else:
                adj[v, u] = -inc.matrix[v, e]
        else:
            adj[u, v] = adj[v, u] = inc.matrix[u, e]
    return adj
