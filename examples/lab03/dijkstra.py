import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, Neighbor, visualize
from graphs.types import AdjacencyList


def _path_subgraph(vertex_count: int, path: list[int]) -> Graph:
    adj: AdjacencyList = [[] for _ in range(vertex_count)]
    for u, v in zip(path, path[1:], strict=False):
        adj[u - 1].append(Neighbor(v, 1))
        adj[v - 1].append(Neighbor(u, 1))
    return Graph(adj, "adjacency_list")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Dijkstra's shortest-path algorithm on a random weighted graph."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input weighted graph file (generates random if omitted).",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=["adjacency_list", "adjacency_matrix", "incidence_matrix"],
        default="adjacency_list",
        help="Representation used in --file.",
    )
    parser.add_argument("--n", type=int, default=12, help="Number of vertices.")
    parser.add_argument(
        "--p", type=float, default=0.35, help="Edge probability for G(n, p)."
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
        graph = Graph.create_random_binomial(args.n, args.p, connected=True)

    print("Weighted adjacency list (vertex: neighbor(weight)):")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip())

    paths = graph.dijkstra(args.start)
    print(f"\nSTART: s = {args.start}")
    for vertex in range(1, graph.vertex_count + 1):
        dist, path = paths[vertex]
        formatted_path = " - ".join(str(v) for v in path)
        print(f"d({vertex}) = {dist} ==> [{formatted_path}]")

    if not args.no_gui:
        target = graph.vertex_count if args.target is None else args.target
        path_subgraph = _path_subgraph(graph.vertex_count, paths[target][1])
        visualize.layered(
            graph,
            start_vertex=args.start,
            title=f"Dijkstra from s={args.start} to t={target}",
            show_weights=True,
            subgraph=path_subgraph,
        )


if __name__ == "__main__":
    main()
