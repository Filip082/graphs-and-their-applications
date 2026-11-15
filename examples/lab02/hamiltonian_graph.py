import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, visualize


def _default_graph() -> Graph:
    return Graph(
        [
            [2, 5, 7],
            [1, 3, 6, 8],
            [2, 4, 7],
            [3, 5, 6, 8],
            [1, 4, 8],
            [2, 4, 7, 8],
            [1, 3, 6],
            [2, 4, 5, 6],
        ],
        "adjacency_list",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a graph is Hamiltonian."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional path to an input graph file.",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=["adjacency_list", "adjacency_matrix", "incidence_matrix"],
        default="adjacency_list",
        help="Representation used in --file.",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="*",
        default=None,
        help="Optional graphic sequence to construct the graph from.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        graph = Graph.from_file(args.file, cast(GraphType, args.data_type))
    elif args.sequence is not None:
        graph = Graph.from_sequence(args.sequence)
    else:
        graph = _default_graph()

    print("Adjacency list:")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(str(v) for v in neighbors)}".rstrip())

    print(f"\nIs Hamiltonian: {graph.is_hamiltonian()}")

    if not args.no_gui:
        visualize.on_circle(graph, title="Hamiltonian check")


if __name__ == "__main__":
    main()
