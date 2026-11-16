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
        elif type in ("incidence_matrix", "directed_incidence_matrix"):
            inc = convert._validate_incidence_matrix(
                cast(MatrixInput, data),
                directed=(type == "directed_incidence_matrix"),
            )
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

        if type in (
            "adjacency_matrix",
            "incidence_matrix",
            "directed_incidence_matrix",
        ):
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
    def _candidate_pairs(
        n: int, directed: bool
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        if directed:
            return np.nonzero(~np.eye(n, dtype=bool))
        return np.triu_indices(n, k=1)

    @staticmethod
    def _random_spanning_tree_indices(
        n: int, rng: np.random.Generator, directed: bool = False
    ) -> list[int]:
        perm = rng.permutation(n)
        indices: list[int] = []
        for i in range(1, n):
            a, b = int(perm[i]), int(perm[int(rng.integers(0, i))])
            if directed:
                u, v = (a, b) if rng.random() < 0.5 else (b, a)
                indices.append(u * (n - 1) + (v - 1 if v > u else v))
            else:
                u, v = min(a, b), max(a, b)
                indices.append(u * (2 * n - u - 1) // 2 + (v - u - 1))
        return indices

    @staticmethod
    def _validate_weight_range(min_weight: int, max_weight: int) -> None:
        if isinstance(min_weight, bool) or not isinstance(min_weight, int):
            raise ValueError(f"min_weight must be an int, got {min_weight!r}")
        if isinstance(max_weight, bool) or not isinstance(max_weight, int):
            raise ValueError(f"max_weight must be an int, got {max_weight!r}")
        if min_weight > max_weight:
            raise ValueError(
                f"min_weight ({min_weight}) must be <= max_weight ({max_weight})"
            )
        if min_weight == 0 and max_weight == 0:
            raise ValueError("Weight range [0, 0] contains no non-zero weights")

    @staticmethod
    def _draw_weights(
        rng: np.random.Generator,
        size: int,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> NDArray[np.int64]:
        Graph._validate_weight_range(min_weight, max_weight)
        if min_weight > 0 or max_weight < 0:
            return rng.integers(min_weight, max_weight + 1, size=size, dtype=np.int64)
        # Skip 0 since 0 denotes no edge in adjacency matrices.
        vals = rng.integers(min_weight, max_weight, size=size, dtype=np.int64)
        vals[vals >= 0] += 1
        return vals

    @staticmethod
    def _random_weighted_matrix(
        n: int,
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        rng: np.random.Generator,
        directed: bool = False,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> "Graph":
        matrix = np.zeros((n, n), dtype=np.int64)
        if len(rows) > 0:
            weights = Graph._draw_weights(rng, len(rows), min_weight, max_weight)
            matrix[rows, cols] = weights
            if not directed:
                matrix[cols, rows] = weights
        return Graph(matrix, "adjacency_matrix")

    @staticmethod
    def create_random_binomial(
        n: int,
        p: float,
        connected: bool = False,
        directed: bool = False,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> "Graph":
        Graph._validate_weight_range(min_weight, max_weight)
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        if not 0.0 <= p <= 1.0:
            raise ValueError("Probability p must be between 0.0 and 1.0")
        if connected and n >= 2 and p == 0.0:
            raise ValueError("Cannot create a connected graph with n >= 2 and p == 0.0")
        if n <= 1:
            return Graph(np.zeros((n, n), dtype=np.int64), "adjacency_matrix")

        rng = np.random.default_rng()
        rows, cols = Graph._candidate_pairs(n, directed)
        mask = rng.random(len(rows)) < p
        if connected:
            mask[Graph._random_spanning_tree_indices(n, rng, directed)] = True
        return Graph._random_weighted_matrix(
            n,
            rows[mask],
            cols[mask],
            rng,
            directed=directed,
            min_weight=min_weight,
            max_weight=max_weight,
        )

    @staticmethod
    def create_random_uniform(
        n: int,
        l: int,  # noqa: E741
        connected: bool = False,
        directed: bool = False,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> "Graph":
        Graph._validate_weight_range(min_weight, max_weight)
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        max_edges = n * (n - 1) if directed else n * (n - 1) // 2
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
        rows, cols = Graph._candidate_pairs(n, directed)
        if connected and n >= 2:
            mask = np.zeros(len(rows), dtype=bool)
            mask[Graph._random_spanning_tree_indices(n, rng, directed)] = True
            if l > n - 1:
                available = np.flatnonzero(~mask)
                mask[rng.choice(available, size=l - (n - 1), replace=False)] = True
            return Graph._random_weighted_matrix(
                n,
                rows[mask],
                cols[mask],
                rng,
                directed=directed,
                min_weight=min_weight,
                max_weight=max_weight,
            )

        chosen = rng.choice(len(rows), size=l, replace=False)
        return Graph._random_weighted_matrix(
            n,
            rows[chosen],
            cols[chosen],
            rng,
            directed=directed,
            min_weight=min_weight,
            max_weight=max_weight,
        )

    def adjacency_list(self) -> AdjacencyList:
        if self.data_type == "adjacency_list":
            return [list(neighbors) for neighbors in cast(AdjacencyList, self.data)]
        if self.data_type == "adjacency_matrix":
            return convert.adjacency_matrix_to_adjacency_list(
                cast(AdjacencyMatrix, self.data)
            )
        return convert.incidence_matrix_to_adjacency_list(
            cast(IncidenceMatrix, self.data),
            directed=(self.data_type == "directed_incidence_matrix"),
        )

    def adjacency_matrix(self) -> AdjacencyMatrix:
        if self.data_type == "adjacency_matrix":
            return cast(AdjacencyMatrix, self.data).copy()
        if self.data_type == "adjacency_list":
            return convert.adjacency_list_to_adjacency_matrix(
                cast(AdjacencyList, self.data)
            )
        return convert.incidence_matrix_to_adjacency_matrix(
            cast(IncidenceMatrix, self.data),
            directed=(self.data_type == "directed_incidence_matrix"),
        )

    def incidence_matrix(self) -> IncidenceMatrix:
        if self.data_type in ("incidence_matrix", "directed_incidence_matrix"):
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

    def _randomize_weights(
        self,
        rng: np.random.Generator | None = None,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> None:
        rng = rng or np.random.default_rng()
        adj = self.adjacency_list()
        pairs = [
            (u, nb.v - 1) for u, nbs in enumerate(adj) for nb in nbs if u < nb.v - 1
        ]
        drawn = Graph._draw_weights(rng, len(pairs), min_weight, max_weight)
        weights = {pair: int(w) for pair, w in zip(pairs, drawn, strict=True)}
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
    def create_random_k_regular(
        n: int,
        k: int,
        connected: bool = False,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> "Graph":
        Graph._validate_weight_range(min_weight, max_weight)
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
            graph._randomize_weights(min_weight=min_weight, max_weight=max_weight)
        return graph

    @staticmethod
    def create_random_eulerian(
        n: int = 6,
        connected: bool = True,
        min_weight: int = 1,
        max_weight: int = 10,
    ) -> "Graph":
        Graph._validate_weight_range(min_weight, max_weight)
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
        graph._randomize_weights(rng, min_weight=min_weight, max_weight=max_weight)
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

    def find_hamiltonian_cycle(self) -> list[int] | None:
        self._require_symmetric("find_hamiltonian_cycle")
        n = self.vertex_count
        if n < 3:
            return None
        adj = self._zero_based_adj()
        if any(len(nbs) < 2 for nbs in adj) or max(_components_of(adj)) > 1:
            return None

        adj_sets = [set(nbs) for nbs in adj]
        visited = [False] * n
        visited[0] = True
        path = [0]

        def backtrack(last: int) -> bool:
            if len(path) == n:
                if 0 in adj_sets[last]:
                    path.append(0)
                    return True
                return False
            for neighbor in adj[last]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    path.append(neighbor)
                    if backtrack(neighbor):
                        return True
                    path.pop()
                    visited[neighbor] = False
            return False

        if backtrack(0):
            return [v + 1 for v in path]
        return None

    def is_hamiltonian(self) -> bool:
        self._require_symmetric("is_hamiltonian")
        return self.find_hamiltonian_cycle() is not None

    def _validate_vertex(self, v: int, name: str = "n") -> None:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"Vertex {name} must be an int, got {v!r}")
        if v < 1 or v > self.vertex_count:
            raise ValueError(
                f"Start vertex {name} must be between 1 and {self.vertex_count}, "
                f"got {v}"
            )

    @staticmethod
    def _build_path(
        dist: Sequence[int | float], prev: Sequence[int | None], v: int
    ) -> list[int]:
        if dist[v] == math.inf:
            return []
        path: list[int] = []
        curr: int | None = v
        while curr is not None:
            path.append(curr + 1)
            curr = prev[curr]
        return path[::-1]

    def dijkstra(self, n: int) -> dict[int, tuple[int | float, list[int]]]:
        self._validate_vertex(n, "n")

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

        return {v + 1: (dist[v], self._build_path(dist, prev, v)) for v in range(count)}

    def kosaraju(self) -> list[list[int]]:
        count = self.vertex_count
        if count == 0:
            return []

        adj = self._zero_based_adj()
        d = [-1] * count
        f = [-1] * count
        t = 0
        order: list[int] = []

        def dfs_visit(v: int) -> None:
            nonlocal t
            t += 1
            d[v] = t
            for u in adj[v]:
                if d[u] == -1:
                    dfs_visit(u)
            t += 1
            f[v] = t
            order.append(v)

        for v in range(count):
            if d[v] == -1:
                dfs_visit(v)

        adj_t: list[list[int]] = [[] for _ in range(count)]
        for u, nbs in enumerate(adj):
            for v in nbs:
                adj_t[v].append(u)

        comp = [-1] * count
        nr = 0

        def components_r(cid: int, v: int) -> None:
            for u in adj_t[v]:
                if comp[u] == -1:
                    comp[u] = cid
                    components_r(cid, u)

        for v in reversed(order):
            if comp[v] == -1:
                nr += 1
                comp[v] = nr
                components_r(nr, v)

        groups: list[list[int]] = [[] for _ in range(nr)]
        for v_idx, cid in enumerate(comp):
            groups[cid - 1].append(v_idx + 1)
        return groups

    @staticmethod
    def _bellman_ford_raw(
        adj: list[list[tuple[int, int]]], start: int
    ) -> tuple[list[int | float], list[int | None]]:
        count = len(adj)
        dist: list[int | float] = [math.inf] * count
        prev: list[int | None] = [None] * count
        dist[start] = 0

        for _ in range(count - 1):
            updated = False
            for u, nbs in enumerate(adj):
                du = dist[u]
                if du == math.inf:
                    continue
                for v, w in nbs:
                    if du + w < dist[v]:
                        dist[v] = du + w
                        prev[v] = u
                        updated = True
            if not updated:
                break

        for u, nbs in enumerate(adj):
            du = dist[u]
            if du == math.inf:
                continue
            for v, w in nbs:
                if du + w < dist[v]:
                    raise ValueError(
                        "Graph contains a negative-weight cycle reachable from source"
                    )

        return dist, prev

    def bellman_ford(self, s: int) -> dict[int, tuple[int | float, list[int]]]:
        self._validate_vertex(s, "s")
        adj = [[(nb.v - 1, nb.w) for nb in nbs] for nbs in self.adjacency_list()]
        dist, prev = self._bellman_ford_raw(adj, s - 1)
        return {
            v + 1: (dist[v], self._build_path(dist, prev, v))
            for v in range(self.vertex_count)
        }

    @staticmethod
    def _dijkstra_distances(
        adj: list[list[tuple[int, int]]], start: int
    ) -> list[int | float]:
        count = len(adj)
        dist: list[int | float] = [math.inf] * count
        dist[start] = 0
        heap: list[tuple[int | float, int]] = [(0, start)]

        while heap:
            du, u = heapq.heappop(heap)
            if du > dist[u]:
                continue
            for v, w in adj[u]:
                cand = du + w
                if cand < dist[v]:
                    dist[v] = cand
                    heapq.heappush(heap, (cand, v))

        return dist

    def johnson(self) -> NDArray[np.float64]:
        count = self.vertex_count
        if count == 0:
            return np.zeros((0, 0), dtype=np.float64)

        adj = [[(nb.v - 1, nb.w) for nb in nbs] for nbs in self.adjacency_list()]
        adj.append([(v, 0) for v in range(count)])

        h, _ = self._bellman_ford_raw(adj, count)
        adj_hat = [
            [(v, w + int(h[u]) - int(h[v])) for v, w in adj[u]] for u in range(count)
        ]

        matrix = np.zeros((count, count), dtype=np.float64)
        for u in range(count):
            d_hat = self._dijkstra_distances(adj_hat, u)
            for v in range(count):
                matrix[u, v] = d_hat[v] - h[u] + h[v]
        return matrix

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
