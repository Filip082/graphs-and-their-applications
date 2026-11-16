import argparse
import math
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from graphs import Graph, GraphType, Neighbor, visualize


def _draw_non_zero_weight(
    rng: np.random.Generator, min_weight: int, max_weight: int
) -> int:
    if min_weight == 0 and max_weight == 0:
        raise ValueError("Weight range [0, 0] contains no non-zero weights")
    if min_weight > 0 or max_weight < 0:
        return int(rng.integers(min_weight, max_weight + 1))
    val = int(rng.integers(min_weight, max_weight))
    return val + 1 if val >= 0 else val


def _assign_random_weights_without_neg_cycle(
    graph: Graph,
    min_weight: int,
    max_weight: int,
    max_attempts: int = 200,
) -> Graph:
    rng = np.random.default_rng()
    base_adj = graph.adjacency_list()
    for _ in range(max_attempts):
        weighted_adj = [
            [
                Neighbor(nb.v, _draw_non_zero_weight(rng, min_weight, max_weight))
                for nb in neighbors
            ]
            for neighbors in base_adj
        ]
        candidate = Graph(weighted_adj, "adjacency_list")
        try:
            candidate.johnson()
        except ValueError:
            continue
        return candidate
    raise RuntimeError(
        "Failed to assign weights without a negative-weight cycle; "
        "try increasing --min-weight."
    )


def _format_distance_matrix(matrix: NDArray[np.float64]) -> str:
    formatted = [
        [
            "inf" if math.isinf(val) else str(int(val) if val.is_integer() else val)
            for val in row
        ]
        for row in matrix
    ]
    width = max((len(cell) for row in formatted for cell in row), default=1)
    return "\n".join(" ".join(f"{cell:>{width}}" for cell in row) for row in formatted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all Lab 04 tasks on a single directed graph."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input weighted digraph file (generates random if omitted).",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=["adjacency_list", "adjacency_matrix", "incidence_matrix"],
        default="adjacency_list",
        help="Representation used in --file.",
    )
    parser.add_argument("--n", type=int, default=7, help="Number of vertices.")
    parser.add_argument(
        "--p",
        type=float,
        default=0.35,
        help="Edge probability for directed G(n, p).",
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=-2,
        help="Minimum random edge weight.",
    )
    parser.add_argument(
        "--max-weight",
        type=int,
        default=10,
        help="Maximum random edge weight.",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=1,
        help="Start vertex for Bellman-Ford (1-based).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        graph = Graph.from_file(
            args.file, cast(GraphType, args.data_type), weighted=True, directed=True
        )
    else:
        graph = Graph.create_random_binomial(
            args.n, args.p, directed=True, min_weight=1, max_weight=1
        )

    print("1. Directed graph adjacency list:")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}. {' '.join(str(nb.v) for nb in neighbors)}".rstrip())

    print("\nAdjacency matrix:")
    print(graph.adjacency_matrix())
    print("\nIncidence matrix:")
    print(graph.incidence_matrix().matrix)

    components = graph.kosaraju()
    print("\n2. Strongly connected components (Kosaraju):")
    for idx, comp in enumerate(components, start=1):
        print(f"  SCC {idx}: {' '.join(str(v) for v in comp)}")

    is_strongly_connected = len(components) == 1 and graph.vertex_count > 0
    if not is_strongly_connected:
        print("\nThe digraph is NOT strongly connected; skipping steps 3 and 4.")
        if not args.no_gui:
            visualize.layered(
                graph,
                title=f"Lab 04 Digraph (n={graph.vertex_count}, disconnected)",
                arrow=True,
            )
        return

    print("\nThe digraph is strongly connected.")
    if args.file is None:
        graph = _assign_random_weights_without_neg_cycle(
            graph, args.min_weight, args.max_weight
        )

    print("\n3. Weighted digraph (vertex: neighbor(weight)):")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip())

    try:
        paths = graph.bellman_ford(args.start)
    except ValueError as exc:
        print(f"\nBellman-Ford failed: {exc}")
        return

    print(f"\nBellman-Ford shortest paths (START: s = {args.start}):")
    for vertex in range(1, graph.vertex_count + 1):
        dist, path = paths[vertex]
        dist_str = "inf" if math.isinf(dist) else str(int(dist))
        print(f"d({vertex}) = {dist_str} ==> [{' - '.join(str(v) for v in path)}]")

    dist_matrix = graph.johnson()
    print("\n4. Distance matrix (Johnson):")
    print(_format_distance_matrix(dist_matrix))

    if not args.no_gui:
        visualize.layered(
            graph,
            title=f"Lab 04 Weighted Digraph (n={graph.vertex_count})",
            show_weights=True,
            arrow=True,
        )


if __name__ == "__main__":
    main()
