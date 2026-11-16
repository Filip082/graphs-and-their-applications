import argparse
import math
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, Neighbor, visualize
from graphs.types import AdjacencyList


def _create_strongly_connected_weighted_digraph(
    n: int,
    p: float,
    min_weight: int,
    max_weight: int,
    max_attempts: int = 500,
) -> Graph:
    for _ in range(max_attempts):
        graph = Graph.create_random_binomial(
            n, p, directed=True, min_weight=min_weight, max_weight=max_weight
        )
        if n > 0 and len(graph.kosaraju()) != 1:
            continue
        try:
            graph.johnson()
        except ValueError:
            continue
        return graph
    raise RuntimeError(
        "Failed to generate a strongly connected weighted digraph without "
        "negative-weight cycles; try increasing --p or --min-weight."
    )


def _path_subgraph(vertex_count: int, path: list[int]) -> Graph:
    adj: AdjacencyList = [[] for _ in range(vertex_count)]
    for u, v in zip(path, path[1:], strict=False):
        adj[u - 1].append(Neighbor(v, 1))
    return Graph(adj, "adjacency_list")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bellman-Ford shortest paths on a strongly connected weighted digraph."
        )
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
        "--start", "-s", type=int, default=1, help="Start vertex (1-based)."
    )
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        default=None,
        help="Target vertex whose shortest path is highlighted (defaults to n).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        graph = Graph.from_file(
            args.file, cast(GraphType, args.data_type), weighted=True
        )
    else:
        graph = _create_strongly_connected_weighted_digraph(
            args.n, args.p, args.min_weight, args.max_weight
        )

    print("Weighted adjacency list (vertex: neighbor(weight)):")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip())

    try:
        paths = graph.bellman_ford(args.start)
    except ValueError as exc:
        print(f"\nBellman-Ford failed: {exc}")
        return

    print(f"\nSTART: s = {args.start}")
    for vertex in range(1, graph.vertex_count + 1):
        dist, path = paths[vertex]
        dist_str = "inf" if math.isinf(dist) else str(int(dist))
        formatted_path = " - ".join(str(v) for v in path)
        print(f"d({vertex}) = {dist_str} ==> [{formatted_path}]")

    if not args.no_gui:
        target = graph.vertex_count if args.target is None else args.target
        path_subgraph = _path_subgraph(graph.vertex_count, paths[target][1])
        visualize.layered(
            graph,
            title=f"Bellman-Ford from s={args.start} to t={target}",
            show_weights=True,
            subgraph=path_subgraph,
            arrow=True,
        )


if __name__ == "__main__":
    main()
