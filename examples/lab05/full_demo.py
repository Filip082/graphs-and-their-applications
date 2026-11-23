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
        description="Run all Lab 05 tasks on a flow network."
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
        "--source", "-s", type=int, default=1, help="Source vertex (1-based)."
    )
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        default=None,
        help="Sink vertex (1-based, defaults to last vertex).",
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
    else:
        network = Graph.create_random_flow_network(args.n)

    source = args.source
    sink = network.vertex_count if args.target is None else args.target

    print(
        f"Flow network (vertices={network.vertex_count}, source={source}, sink={sink}):"
    )
    print("\nAdjacency list (vertex: neighbor(capacity)):")
    print(_format_weighted_adjacency_list(network.adjacency_list()))
    print("\nCapacity matrix:")
    print(network.adjacency_matrix())

    max_flow, flow_graph = network.ford_fulkerson(source, sink)
    flow_mat = flow_graph.adjacency_matrix()
    edge_labels = {
        (u, nb.v): f"{int(flow_mat[u - 1, nb.v - 1])}/{nb.w}"
        for u, nbs in enumerate(network.adjacency_list(), start=1)
        for nb in nbs
    }

    print(f"\n|f_max| = {max_flow}")
    print("\nEdge flows f(u, v)/c(u, v):")
    for u, nbs in enumerate(network.adjacency_list(), start=1):
        if nbs:
            edges_str = " ".join(f"{u}->{nb.v}: {edge_labels[(u, nb.v)]}" for nb in nbs)
            print(f"  {edges_str}")

    if not args.no_gui:
        visualize.layered(
            network,
            start_vertex=source,
            title=f"Flow network (|f_max| = {max_flow})",
            show_weights=True,
            subgraph=flow_graph,
            arrow=True,
            edge_labels=edge_labels,
        )


if __name__ == "__main__":
    main()
