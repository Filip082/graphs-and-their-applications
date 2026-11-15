import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, visualize


def _default_graph() -> Graph:
    return Graph(
        [
            [1, 4, 6],
            [0, 2, 5, 7],
            [1, 3, 6],
            [2, 4, 5, 7],
            [0, 3, 7],
            [1, 3, 6, 7],
            [0, 2, 5],
            [1, 3, 4, 5],
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
    for u, neighbors in enumerate(graph.adjacency_list()):
        print(f"{u}: {' '.join(str(v) for v in neighbors)}".rstrip())

    print(f"\nIs Hamiltonian: {graph.is_hamiltonian()}")

    if not args.no_gui:
        visualize.on_circle(graph, title="Hamiltonian check")


if __name__ == "__main__":
    main()
