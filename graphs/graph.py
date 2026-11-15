import sys
from collections import Counter, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from graphs import convert, utils
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

    def _set_from_adjacency_list(self, adj: AdjacencyList) -> None:
        if self.data_type == "adjacency_list":
            self.data = adj
        elif self.data_type == "adjacency_matrix":
            self.data = convert.adjacency_list_to_adjacency_matrix(adj)
        else:
            self.data = convert.adjacency_list_to_incidence_matrix(adj)

    def _components(self) -> list[int]:
        adj = self.adjacency_list()
        n = self.vertex_count
        if n >= 900:
            sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 100))
        comp = [-1] * n
        nr = 0

        def components_r(component_id: int, v: int) -> None:
            for u in adj[v]:
                if comp[u] == -1:
                    comp[u] = component_id
                    components_r(component_id, u)

        for v in range(n):
            if comp[v] == -1:
                nr += 1
                comp[v] = nr
                components_r(nr, v)
        return comp

    @staticmethod
    def from_sequence(sequence: Sequence[int] | NDArray[np.integer]) -> "Graph":
        if not utils.is_graphic_sequence(sequence):
            raise ValueError("Sequence is not a graphic sequence")

        items = [[int(deg), idx] for idx, deg in enumerate(sequence)]
        adj: AdjacencyList = [[] for _ in range(len(items))]

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

        return Graph(adj, "adjacency_list")

    def randomize(self, n: int) -> None:
        if n < 0:
            raise ValueError("Number of randomization steps n must be non-negative")
        if n == 0:
            return

        adj = self.adjacency_list()
        adj_sets = [set(neighbors) for neighbors in adj]
        edges: list[tuple[int, int]] = [
            (u, v) for u, neighbors in enumerate(adj) for v in neighbors if u < v
        ]
        if len(edges) < 2:
            return

        rng = np.random.default_rng()

        def try_swap(i: int, j: int) -> bool:
            a, b = edges[i]
            c, d = edges[j]
            orientations = [(c, d), (d, c)] if rng.random() < 0.5 else [(d, c), (c, d)]
            for x, y in orientations:
                if (
                    len({a, b, x, y}) == 4
                    and y not in adj_sets[a]
                    and x not in adj_sets[b]
                ):
                    adj_sets[a].remove(b)
                    adj_sets[b].remove(a)
                    adj_sets[x].remove(y)
                    adj_sets[y].remove(x)
                    adj_sets[a].add(y)
                    adj_sets[y].add(a)
                    adj_sets[b].add(x)
                    adj_sets[x].add(b)
                    edges[i] = (min(a, y), max(a, y))
                    edges[j] = (min(b, x), max(b, x))
                    return True
            return False

        m = len(edges)
        max_attempts = 100
        for _ in range(n):
            swapped = False
            for _ in range(max_attempts):
                i = int(rng.integers(0, m))
                j = int(rng.integers(0, m - 1))
                if j >= i:
                    j += 1
                if try_swap(i, j):
                    swapped = True
                    break
            if not swapped:
                found_any = False
                for i in range(m):
                    for j in range(i + 1, m):
                        if try_swap(i, j):
                            found_any = True
                            break
                    if found_any:
                        break
                if not found_any:
                    break

        self._set_from_adjacency_list([sorted(neighbors) for neighbors in adj_sets])

    @staticmethod
    def create_random_k_regular(n: int, k: int) -> "Graph":
        if n < 0 or k < 0:
            raise ValueError("n and k must be non-negative")
        if n == 0 and k > 0:
            raise ValueError(
                "Cannot create a k-regular graph with 0 vertices and k > 0"
            )
        if n > 0 and (k >= n or (n * k) % 2 != 0):
            raise ValueError(f"Cannot create {k}-regular simple graph on {n} vertices")

        graph = Graph.from_sequence([k] * n)
        if n > 0 and k > 0:
            graph.randomize(max(10, n * k * 2))
        return graph

    @staticmethod
    def create_random_eulerian(n: int = 6) -> "Graph":
        if n < 0:
            raise ValueError("Number of vertices n must be non-negative")
        if n == 2:
            raise ValueError(
                "Cannot create a connected Eulerian simple graph with 2 vertices"
            )
        if n == 0:
            return Graph([], "adjacency_list")
        if n == 1:
            return Graph([[]], "adjacency_list")

        max_even = n - 1 if n % 2 == 1 else n - 2
        rng = np.random.default_rng()
        seq: list[int] = [2] * n
        for _ in range(50):
            candidate = (rng.integers(1, max_even // 2 + 1, size=n) * 2).tolist()
            if utils.is_graphic_sequence(candidate):
                seq = candidate
                break

        graph = Graph.from_sequence(seq)
        graph.randomize(n * 4)

        comp = graph._components()
        while max(comp) > 1:
            adj_sets = [set(neighbors) for neighbors in graph.adjacency_list()]
            edge_in_c1: tuple[int, int] | None = None
            edge_in_other: tuple[int, int] | None = None
            for u, neighbors in enumerate(adj_sets):
                for v in neighbors:
                    if u < v:
                        if comp[u] == 1 and edge_in_c1 is None:
                            edge_in_c1 = (u, v)
                        elif comp[u] != 1 and edge_in_other is None:
                            edge_in_other = (u, v)
                    if edge_in_c1 is not None and edge_in_other is not None:
                        break
                if edge_in_c1 is not None and edge_in_other is not None:
                    break
            if edge_in_c1 is None or edge_in_other is None:
                break
            a, b = edge_in_c1
            c, d = edge_in_other
            adj_sets[a].remove(b)
            adj_sets[b].remove(a)
            adj_sets[c].remove(d)
            adj_sets[d].remove(c)
            adj_sets[a].add(d)
            adj_sets[d].add(a)
            adj_sets[b].add(c)
            adj_sets[c].add(b)
            graph._set_from_adjacency_list([sorted(s) for s in adj_sets])
            comp = graph._components()

        return graph

    def largest_connected_component(self) -> list[int]:
        if self.vertex_count == 0:
            return []
        comp = self._components()
        counts = Counter(comp)
        largest_id = max(counts, key=lambda cid: counts[cid])
        return [v for v, cid in enumerate(comp) if cid == largest_id]

    def find_euler_cycle(self) -> list[int] | None:
        if self.vertex_count == 0:
            return []
        adj = self.adjacency_list()
        if any(len(neighbors) % 2 != 0 for neighbors in adj):
            return None
        comp = self._components()
        if max(comp) > 1:
            return None
        if all(len(neighbors) == 0 for neighbors in adj):
            return [0]

        rem = [sorted(neighbors) for neighbors in adj]

        def reachable_count(start: int) -> int:
            visited = {start}
            queue = deque([start])
            while queue:
                curr = queue.popleft()
                for nxt in rem[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            return len(visited)

        def is_bridge(u: int, v: int) -> bool:
            before = reachable_count(u)
            rem[u].remove(v)
            rem[v].remove(u)
            after = reachable_count(u)
            rem[u].append(v)
            rem[v].append(u)
            rem[u].sort()
            rem[v].sort()
            return after < before

        u = 0
        cycle = [u]
        while rem[u]:
            if len(rem[u]) == 1:
                v = rem[u][0]
            else:
                v = rem[u][0]
                for candidate in rem[u]:
                    if not is_bridge(u, candidate):
                        v = candidate
                        break
            rem[u].remove(v)
            rem[v].remove(u)
            u = v
            cycle.append(u)

        return cycle

    def is_hamiltonian(self) -> bool:
        n = self.vertex_count
        if n < 3:
            return False
        adj = self.adjacency_list()
        if any(len(neighbors) < 2 for neighbors in adj):
            return False
        comp = self._components()
        if max(comp) > 1:
            return False

        adj_sets = [set(neighbors) for neighbors in adj]
        stack = [0]
        visited = [False] * n
        visited[0] = True

        def backtrack() -> bool:
            if len(stack) == n:
                return stack[0] in adj_sets[stack[-1]]
            last = stack[-1]
            for neighbor in adj[last]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
                    if backtrack():
                        return True
                    stack.pop()
                    visited[neighbor] = False
            return False

        return backtrack()
