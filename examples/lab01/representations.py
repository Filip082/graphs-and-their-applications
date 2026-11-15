import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, visualize

DEFAULT_FILE = Path(__file__).parent / "data" / "graph_adj_matrix.txt"


def _format_adjacency_list(adj_list: list[list[int]]) -> str:
    return "\n".join(
        f"{u}: {' '.join(str(v) for v in neighbors)}".rstrip()
        for u, neighbors in enumerate(adj_list)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a graph in one representation and output the other two."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Path to the input graph file.",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=["adjacency_list", "adjacency_matrix", "incidence_matrix"],
        default="adjacency_matrix",
        help="Representation used in the input file.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    data_type = cast(GraphType, args.data_type)
    graph = Graph.from_file(args.file, data_type)

    if data_type != "adjacency_list":
        print("Adjacency list:")
        print(_format_adjacency_list(graph.adjacency_list()))
        print()

    if data_type != "adjacency_matrix":
        print("Adjacency matrix:")
        print(graph.adjacency_matrix())
        print()

    if data_type != "incidence_matrix":
        print("Incidence matrix:")
        print(graph.incidence_matrix())
        print()

    if not args.no_gui:
        visualize.on_circle(graph, title=f"Graph ({data_type})")


if __name__ == "__main__":
    main()
