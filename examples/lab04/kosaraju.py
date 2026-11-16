import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, visualize
from graphs.types import AdjacencyList


def _format_adjacency_list(adj_list: AdjacencyList) -> str:
    return "\n".join(
        f"{u}: {' '.join(str(neighbor.v) for neighbor in neighbors)}".rstrip()
        for u, neighbors in enumerate(adj_list, start=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find strongly connected components using Kosaraju's algorithm."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input digraph file (generates random if omitted).",
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
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        graph = Graph.from_file(
            args.file, cast(GraphType, args.data_type), directed=True
        )
        title = f"Kosaraju ({args.file.name})"
    else:
        graph = Graph.create_random_binomial(
            args.n, args.p, directed=True, min_weight=1, max_weight=1
        )
        title = f"Kosaraju G({args.n}, {args.p})"

    print("Adjacency list:")
    print(_format_adjacency_list(graph.adjacency_list()))

    components = graph.kosaraju()
    print("\nStrongly connected components (Kosaraju):")
    for idx, comp in enumerate(components, start=1):
        print(f"  SCC {idx}: {' '.join(str(v) for v in comp)}")

    if len(components) == 1 and graph.vertex_count > 0:
        print("\nThe digraph is strongly connected.")
    else:
        print("\nThe digraph is NOT strongly connected.")

    if not args.no_gui:
        visualize.layered(graph, title=title, arrow=True)


if __name__ == "__main__":
    main()
