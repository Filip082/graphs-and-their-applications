import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a graph (from file or random) is Hamiltonian."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input graph file (generates a random graph if omitted).",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=["adjacency_list", "adjacency_matrix", "incidence_matrix"],
        default="adjacency_list",
        help="Representation used in --file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Number of vertices for random graph generation.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Edge probability for random graph generation.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        graph = Graph.from_file(args.file, cast(GraphType, args.data_type))
    else:
        graph = Graph.create_random_binomial(args.n, args.p)

    print("Adjacency list:")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(str(neighbor.v) for neighbor in neighbors)}".rstrip())

    print(f"\nIs Hamiltonian: {graph.is_hamiltonian()}")

    if not args.no_gui:
        visualize.circular(graph, title="Hamiltonian check")


if __name__ == "__main__":
    main()
