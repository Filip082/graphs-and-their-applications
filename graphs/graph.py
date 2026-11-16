import heapq
import math
from collections import deque
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from graphs import convert, utils
from graphs.types import (
    AdjacencyList,
    AdjacencyListInput,
    AdjacencyMatrix,
    GraphType,
    IncidenceMatrix,
    MatrixInput,
    Neighbor,
)


def _components_of(adj: Sequence[Collection[int]]) -> list[int]:
    comp = [0] * len(adj)
    cid = 0
    for start in range(len(adj)):
        if comp[start] == 0:
            cid += 1
            comp[start] = cid
            stack = [start]
            while stack:
                v = stack.pop()
                for u in adj[v]:
                    if comp[u] == 0:
                        comp[u] = cid
                        stack.append(u)
    return comp


def _swap_edges(adj_sets: list[set[int]], a: int, b: int, c: int, d: int) -> None:
    adj_sets[a].remove(b)
    adj_sets[b].remove(a)
    adj_sets[c].remove(d)
    adj_sets[d].remove(c)
    adj_sets[a].add(d)
    adj_sets[d].add(a)
    adj_sets[b].add(c)
    adj_sets[c].add(b)


class Graph:
    vertex_count: int
    data_type: GraphType
    data: AdjacencyList | AdjacencyMatrix | IncidenceMatrix

    def __init__(
        self,
        data: AdjacencyListInput | MatrixInput,
        type: GraphType,
    ) -> None:
        if type == "adjacency_list":
            if isinstance(data, np.ndarray):
                raise ValueError("Adjacency list must be a sequence, not an ndarray")
            adj = convert._validate_adjacency_list(cast(AdjacencyListInput, data))
            self.vertex_count = len(adj)
            self.data = adj
        elif type == "adjacency_matrix":
            mat = convert._validate_adjacency_matrix(cast(MatrixInput, data))
            self.vertex_count = int(mat.shape[0])
            self.data = mat
        elif type == "incidence_matrix":
            inc = convert._validate_incidence_matrix(cast(MatrixInput, data))
            self.vertex_count = int(inc.shape[0])
            self.data = inc
        self.data_type = type

    @staticmethod
    def from_file(
        path: str | Path,
        type: GraphType,
        weighted: bool = False,
    ) -> "Graph":
        text = Path(path).read_text(encoding="utf-8")
        if type == "adjacency_list":
            parsed: AdjacencyList = []
            for line in text.splitlines():
                tokens = (line.split(":", 1)[1] if ":" in line else line).split()
                if weighted:
                    if len(tokens) % 2:
                        raise ValueError(
                            "Weighted adjacency list line must have vertex-weight pairs"
                        )
                    parsed.append(
                        [
                            Neighbor(int(tokens[i]), int(tokens[i + 1]))
                            for i in range(0, len(tokens), 2)
                        ]
                    )
                else:
                    parsed.append([Neighbor(int(tok), 1) for tok in tokens])
            return Graph(parsed, type)

        if type in ("adjacency_matrix", "incidence_matrix"):
            rows = [
                [int(tok) for tok in line.split()]
                for line in text.splitlines()
                if line.strip()
            ]
            if not rows:
                return Graph(np.zeros((0, 0), dtype=np.int64), type)
            if len({len(row) for row in rows}) != 1:
                raise ValueError("All matrix rows in file must have equal length")
            return Graph(np.array(rows, dtype=np.int64), type)

        raise ValueError(f"Unsupported graph data type: {type!r}")

    @staticmethod
    def _random_spanning_tree_indices(n: int, rng: np.random.Generator) -> list[int]:
        perm = rng.permutation(n)
        indices: list[int] = []
        for i in range(1, n):
            a, b = int(perm[i]), int(perm[int(rng.integers(0, i))])
            u, v = min(a, b), max(a, b)
            indices.append(u * (2 * n - u - 1) // 2 + (v - u - 1))
        return indices

    @staticmethod
    def _random_weighted_matrix(
        n: int,
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        rng: np.random.Generator,
    ) -> "Graph":
        matrix = np.zeros((n, n), dtype=np.int64)
        if len(rows) > 0:
            weights = rng.integers(1, 11, size=len(rows), dtype=np.int64)
            matrix[rows, cols] = weights
            matrix[cols, rows] = weights
        return Graph(matrix, "adjacency_matrix")

    @staticmethod
    def create_random_binomial(n: int, p: float, connected: bool = False) -> "Graph":
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        if not 0.0 <= p <= 1.0:
            raise ValueError("Probability p must be between 0.0 and 1.0")
        if connected and n >= 2 and p == 0.0:
            raise ValueError("Cannot create a connected graph with n >= 2 and p == 0.0")
        if n <= 1:
            return Graph(np.zeros((n, n), dtype=np.int64), "adjacency_matrix")

        rng = np.random.default_rng()
        rows, cols = np.triu_indices(n, k=1)
        mask = rng.random(len(rows)) < p
        if connected:
            mask[Graph._random_spanning_tree_indices(n, rng)] = True
        return Graph._random_weighted_matrix(n, rows[mask], cols[mask], rng)

    @staticmethod
    def create_random_uniform(
        n: int,
        l: int,  # noqa: E741
        connected: bool = False,
    ) -> "Graph":
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        max_edges = n * (n - 1) // 2
        if l < 0 or l > max_edges:
            raise ValueError(f"Number of edges l must be between 0 and {max_edges}")
        if connected and n >= 2 and l < n - 1:
            raise ValueError(
                f"Cannot create a connected graph on {n} vertices "
                f"with fewer than {n - 1} edges"
            )
        if l == 0:
            return Graph(np.zeros((n, n), dtype=np.int64), "adjacency_matrix")

        rng = np.random.default_rng()
        rows, cols = np.triu_indices(n, k=1)
        if connected and n >= 2:
            mask = np.zeros(len(rows), dtype=bool)
            mask[Graph._random_spanning_tree_indices(n, rng)] = True
            if l > n - 1:
                available = np.flatnonzero(~mask)
                mask[rng.choice(available, size=l - (n - 1), replace=False)] = True
            return Graph._random_weighted_matrix(n, rows[mask], cols[mask], rng)

        chosen = rng.choice(len(rows), size=l, replace=False)
        return Graph._random_weighted_matrix(n, rows[chosen], cols[chosen], rng)

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

    def _require_symmetric(self, operation: str) -> None:
        mat = self.adjacency_matrix()
        if not np.array_equal(mat, mat.T):
            raise ValueError(
                f"{operation} is only supported for symmetric (undirected) graphs"
            )

    def _zero_based_adj(self) -> list[list[int]]:
        return [
            [neighbor.v - 1 for neighbor in neighbors]
            for neighbors in self.adjacency_list()
        ]

    def _set_from_adjacency_list(self, adj: AdjacencyList) -> None:
        if self.data_type == "adjacency_list":
            self.data = adj
        elif self.data_type == "adjacency_matrix":
            self.data = convert.adjacency_list_to_adjacency_matrix(adj)
        else:
            self.data = convert.adjacency_list_to_incidence_matrix(adj)

    def _set_from_zero_based_adj(self, adj: list[list[int]]) -> None:
        self._set_from_adjacency_list(
            [[Neighbor(v + 1, 1) for v in neighbors] for neighbors in adj]
        )

    def _randomize_weights(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        adj = self.adjacency_list()
        weights = {
            (u, nb.v - 1): int(rng.integers(1, 11))
            for u, nbs in enumerate(adj)
            for nb in nbs
            if u < nb.v - 1
        }
        self._set_from_adjacency_list(
            [
                [
                    Neighbor(nb.v, weights[(min(u, nb.v - 1), max(u, nb.v - 1))])
                    for nb in nbs
                ]
                for u, nbs in enumerate(adj)
            ]
        )

    @staticmethod
    def from_sequence(sequence: Sequence[int] | NDArray[np.integer]) -> "Graph":
        if not utils.is_graphic_sequence(sequence):
            raise ValueError("Sequence is not a graphic sequence")

        items = [[int(deg), idx] for idx, deg in enumerate(sequence)]
        adj: list[list[int]] = [[] for _ in range(len(items))]
        while items:
            items.sort(key=lambda pair: pair[0], reverse=True)
            deg, u = items[0]
            if deg == 0:
                break
            items[0][0] = 0
            for i in range(1, deg + 1):
                v = items[i][1]
                items[i][0] -= 1
                adj[u].append(v)
                adj[v].append(u)

        return Graph(
            [[Neighbor(v + 1, 1) for v in neighbors] for neighbors in adj],
            "adjacency_list",
        )

    def randomize(self, n: int) -> None:
        if n < 0:
            raise ValueError("Number of randomization steps n must be non-negative")
        self._require_symmetric("randomize")
        if n == 0:
            return

        adj = self._zero_based_adj()
        adj_sets = [set(neighbors) for neighbors in adj]
        edges = [(u, v) for u, nbs in enumerate(adj) for v in nbs if u < v]
        m = len(edges)
        if m < 2:
            return

        rng = np.random.default_rng()

        def try_swap(i: int, j: int) -> bool:
            a, b = edges[i]
            c, d = edges[j]
            orientations = ((c, d), (d, c)) if rng.random() < 0.5 else ((d, c), (c, d))
            for x, y in orientations:
                if (
                    len({a, b, x, y}) == 4
                    and y not in adj_sets[a]
                    and x not in adj_sets[b]
                ):
                    _swap_edges(adj_sets, a, b, x, y)
                    edges[i] = (min(a, y), max(a, y))
                    edges[j] = (min(b, x), max(b, x))
                    return True
            return False

        for _ in range(n):
            for _ in range(100):
                i = int(rng.integers(0, m))
                j = int(rng.integers(0, m - 1))
                if try_swap(i, j + int(j >= i)):
                    break
            else:
                if not any(try_swap(i, j) for i in range(m) for j in range(i + 1, m)):
                    break

        self._set_from_zero_based_adj([sorted(neighbors) for neighbors in adj_sets])

    def _connect_components(self) -> None:
        adj_sets = [set(neighbors) for neighbors in self._zero_based_adj()]
        comp = _components_of(adj_sets)
        changed = False

        def try_merge(num_comp: int) -> list[int] | None:
            c1 = [
                (u, v)
                for u, nbs in enumerate(adj_sets)
                for v in nbs
                if u < v and comp[u] == 1
            ]
            other = [
                (u, v)
                for u, nbs in enumerate(adj_sets)
                for v in nbs
                if u < v and comp[u] != 1
            ]
            for a, b in c1:
                for c, d in other:
                    _swap_edges(adj_sets, a, b, c, d)
                    new_comp = _components_of(adj_sets)
                    if max(new_comp) < num_comp:
                        return new_comp
                    _swap_edges(adj_sets, a, d, c, b)
            return None

        while (num_comp := max(comp, default=0)) > 1:
            new_comp = try_merge(num_comp)
            if new_comp is None:
                break
            comp = new_comp
            changed = True

        if changed:
            self._set_from_zero_based_adj([sorted(s) for s in adj_sets])

    @staticmethod
    def create_random_k_regular(n: int, k: int, connected: bool = False) -> "Graph":
        if n < 0 or k < 0 or (n == 0 and k > 0) or (n > 0 and (k >= n or n * k % 2)):
            raise ValueError(f"Cannot create {k}-regular simple graph on {n} vertices")
        if connected and ((n >= 2 and k == 0) or (n >= 3 and k == 1)):
            raise ValueError(
                f"Cannot create a connected {k}-regular graph on {n} vertices"
            )

        graph = Graph.from_sequence([k] * n)
        if n > 0 and k > 0:
            graph.randomize(max(10, n * k * 2))
            if connected:
                graph._connect_components()
            graph._randomize_weights()
        return graph

    @staticmethod
    def create_random_eulerian(n: int = 6, connected: bool = True) -> "Graph":
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        if n == 2:
            raise ValueError(
                "Cannot create a connected Eulerian simple graph with 2 vertices"
            )
        if n <= 1:
            return Graph([[] for _ in range(n)], "adjacency_list")

        max_even = n - 1 if n % 2 == 1 else n - 2
        rng = np.random.default_rng()
        seq = [2] * n
        for _ in range(50):
            candidate = (rng.integers(1, max_even // 2 + 1, size=n) * 2).tolist()
            if utils.is_graphic_sequence(candidate):
                seq = candidate
                break

        graph = Graph.from_sequence(seq)
        graph.randomize(n * 4)
        if connected:
            graph._connect_components()
        graph._randomize_weights(rng)
        return graph

    def connected_components(self) -> list[list[int]]:
        self._require_symmetric("connected_components")
        if self.vertex_count == 0:
            return []
        comp = _components_of(self._zero_based_adj())
        groups: list[list[int]] = [[] for _ in range(max(comp))]
        for v_idx, cid in enumerate(comp):
            groups[cid - 1].append(v_idx + 1)
        return groups

    def largest_connected_component(self) -> list[int]:
        return max(self.connected_components(), key=len, default=[])

    def find_euler_cycle(self) -> list[int] | None:
        self._require_symmetric("find_euler_cycle")
        if self.vertex_count == 0:
            return []
        adj = self._zero_based_adj()
        if any(len(nbs) % 2 for nbs in adj) or max(_components_of(adj)) > 1:
            return None
        if all(not nbs for nbs in adj):
            return [1]

        def is_bridge(u: int, v: int) -> bool:
            visited = {u}
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    if (curr, nxt) in ((u, v), (v, u)):
                        continue
                    if nxt == v:
                        return False
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            return True

        u = 0
        cycle = [u]
        while adj[u]:
            if len(adj[u]) == 1:
                v = adj[u][0]
            else:
                v = next((c for c in adj[u] if not is_bridge(u, c)), adj[u][0])
            adj[u].remove(v)
            adj[v].remove(u)
            u = v
            cycle.append(u)

        return [v + 1 for v in cycle]

    def is_hamiltonian(self) -> bool:
        self._require_symmetric("is_hamiltonian")
        n = self.vertex_count
        if n < 3:
            return False
        adj = self._zero_based_adj()
        if any(len(nbs) < 2 for nbs in adj) or max(_components_of(adj)) > 1:
            return False

        adj_sets = [set(nbs) for nbs in adj]
        visited = [False] * n
        visited[0] = True

        def backtrack(last: int, depth: int) -> bool:
            if depth == n:
                return 0 in adj_sets[last]
            for neighbor in adj[last]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if backtrack(neighbor, depth + 1):
                        return True
                    visited[neighbor] = False
            return False

        return backtrack(0, 1)

    def dijkstra(self, n: int) -> dict[int, tuple[int | float, list[int]]]:
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"Vertex n must be an int, got {n!r}")
        if n < 1 or n > self.vertex_count:
            raise ValueError(
                f"Start vertex n must be between 1 and {self.vertex_count}, got {n}"
            )

        adj = self.adjacency_list()
        if any(nb.w < 0 for nbs in adj for nb in nbs):
            raise ValueError("Dijkstra's algorithm requires non-negative edge weights")

        count = self.vertex_count
        dist: list[int | float] = [math.inf] * count
        prev: list[int | None] = [None] * count
        dist[n - 1] = 0
        ready = [False] * count

        for _ in range(count):
            u = -1
            best_dist = math.inf
            for candidate in range(count):
                if not ready[candidate] and dist[candidate] < best_dist:
                    best_dist = dist[candidate]
                    u = candidate
            if u == -1:
                break
            ready[u] = True

            for neighbor in adj[u]:
                v = neighbor.v - 1
                if not ready[v] and dist[u] + neighbor.w < dist[v]:
                    dist[v] = dist[u] + neighbor.w
                    prev[v] = u

        def build_path(v: int) -> list[int]:
            if dist[v] == math.inf:
                return []
            path: list[int] = []
            curr: int | None = v
            while curr is not None:
                path.append(curr + 1)
                curr = prev[curr]
            return path[::-1]

        return {v + 1: (dist[v], build_path(v)) for v in range(count)}

    def distance_matrix(self) -> NDArray[np.int64]:
        count = self.vertex_count
        matrix = np.zeros((count, count), dtype=np.int64)
        for u in range(1, count + 1):
            paths = self.dijkstra(u)
            row = [paths[v][0] for v in range(1, count + 1)]
            if any(math.isinf(d) for d in row):
                raise ValueError(
                    "Cannot compute distance matrix for a disconnected graph"
                )
            matrix[u - 1] = row
        return matrix

    def _center_by(self, metric: Literal["sum", "max"]) -> tuple[int, int]:
        if self.vertex_count == 0:
            raise ValueError("Cannot find center of an empty graph")
        dist = self.distance_matrix()
        scores = dist.sum(axis=1) if metric == "sum" else dist.max(axis=1)
        idx = int(np.argmin(scores))
        return idx + 1, int(scores[idx])

    def center(self) -> tuple[int, int]:
        return self._center_by("sum")

    def minimax(self) -> tuple[int, int]:
        return self._center_by("max")

    @staticmethod
    def _graph_from_weighted_edges(
        count: int, edges: list[tuple[int, int, int]]
    ) -> "Graph":
        mst_adj: AdjacencyList = [[] for _ in range(count)]
        for u, v, w in edges:
            mst_adj[u].append(Neighbor(v + 1, w))
            mst_adj[v].append(Neighbor(u + 1, w))
        return Graph(mst_adj, "adjacency_list")

    def prim(self) -> "Graph":
        self._require_symmetric("prim")
        count = self.vertex_count
        if count <= 1:
            return Graph._graph_from_weighted_edges(count, [])

        adj = self.adjacency_list()
        in_tree = [False] * count
        in_tree[0] = True
        tree_count = 1
        heap: list[tuple[int, int, int]] = [(nb.w, 0, nb.v - 1) for nb in adj[0]]
        heapq.heapify(heap)
        mst_edges: list[tuple[int, int, int]] = []

        while heap and tree_count < count:
            w, u, v = heapq.heappop(heap)
            if in_tree[v]:
                continue
            in_tree[v] = True
            tree_count += 1
            mst_edges.append((u, v, w))
            for nb in adj[v]:
                nxt = nb.v - 1
                if not in_tree[nxt]:
                    heapq.heappush(heap, (nb.w, v, nxt))

        if tree_count < count:
            raise ValueError("Cannot find spanning tree for a disconnected graph")
        return Graph._graph_from_weighted_edges(count, mst_edges)

    def kruskal(self) -> "Graph":
        self._require_symmetric("kruskal")
        count = self.vertex_count
        if count <= 1:
            return Graph._graph_from_weighted_edges(count, [])

        edges = sorted(
            (nb.w, u, nb.v - 1)
            for u, nbs in enumerate(self.adjacency_list())
            for nb in nbs
            if u < nb.v - 1
        )
        parent = list(range(count))
        rank = [0] * count

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        mst_edges: list[tuple[int, int, int]] = []
        for w, u, v in edges:
            ra, rb = find(u), find(v)
            if ra == rb:
                continue
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            mst_edges.append((u, v, w))
            if len(mst_edges) == count - 1:
                break

        if len(mst_edges) < count - 1:
            raise ValueError("Cannot find spanning tree for a disconnected graph")
        return Graph._graph_from_weighted_edges(count, mst_edges)
