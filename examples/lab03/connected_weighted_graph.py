import argparse

from graphs import Graph, visualize
from graphs.types import AdjacencyList


def _format_weighted_adj(adj: AdjacencyList) -> str:
    return "\n".join(
        f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip()
        for u, neighbors in enumerate(adj, start=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a connected random undirected graph with edge weights."
    )
    parser.add_argument("--n", type=int, default=12, help="Number of vertices.")
    parser.add_argument(
        "--p",
        type=float,
        default=0.35,
        help="Edge probability for G(n, p) (ignored if --l is set).",
    )
    parser.add_argument(
        "--l",
        type=int,
        default=None,
        help="Exact number of edges for G(n, l) instead of G(n, p).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.l is not None:
        graph = Graph.create_random_uniform(args.n, args.l, connected=True)
        title = f"Connected weighted G({args.n}, {args.l})"
    else:
        graph = Graph.create_random_binomial(args.n, args.p, connected=True)
        title = f"Connected weighted G({args.n}, {args.p})"

    print("Weighted adjacency list (vertex: neighbor(weight)):")
    print(_format_weighted_adj(graph.adjacency_list()))
    print("\nWeighted adjacency matrix:")
    print(graph.adjacency_matrix())

    if not args.no_gui:
        visualize.layered(graph, title=title)


if __name__ == "__main__":
    main()
