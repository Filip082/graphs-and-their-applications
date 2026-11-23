import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, visualize
from graphs.types import AdjacencyList


def _format_weighted_adjacency_list(adj_list: AdjacencyList) -> str:
    return "\n".join(
        f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip()
        for u, neighbors in enumerate(adj_list, start=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a random flow network and display its capacities."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input flow network file (generates random if omitted).",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=[
            "adjacency_list",
            "adjacency_matrix",
            "incidence_matrix",
            "directed_incidence_matrix",
        ],
        default="adjacency_list",
        help="Representation used in --file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of intermediate layers (N >= 2, recommended <= 4).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        network = Graph.from_file(
            args.file, cast(GraphType, args.data_type), weighted=True
        )
        title = f"Flow network ({args.file.name})"
    else:
        network = Graph.create_random_flow_network(args.n)
        title = f"Random flow network (N={args.n})"

    print("Adjacency list (vertex: neighbor(capacity)):")
    print(_format_weighted_adjacency_list(network.adjacency_list()))
    print("\nCapacity matrix:")
    print(network.adjacency_matrix())

    if not args.no_gui:
        visualize.layered(
            network,
            start_vertex=1,
            title=title,
            show_weights=True,
            arrow=True,
        )


if __name__ == "__main__":
    main()
