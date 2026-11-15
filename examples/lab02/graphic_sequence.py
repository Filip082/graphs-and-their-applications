import argparse

from graphs import Graph, utils, visualize


def _print_adjacency_list(graph: Graph) -> None:
    for u, neighbors in enumerate(graph.adjacency_list()):
        print(f"{u}: {' '.join(str(v) for v in neighbors)}".rstrip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check if a sequence is graphic, build a graph, "
            "optionally randomize edges, and find the largest connected component."
        )
    )
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="*",
        default=[4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2],
        help="Degree sequence to check and realize.",
    )
    parser.add_argument(
        "--randomize",
        type=int,
        default=10,
        help="Number of edge-swap randomization steps (0 to skip).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    is_graphic = utils.is_graphic_sequence(args.sequence)
    print(f"Sequence {args.sequence} is graphic: {is_graphic}")
    if not is_graphic:
        return

    graph = Graph.from_sequence(args.sequence)
    print("\nConstructed graph (adjacency list):")
    _print_adjacency_list(graph)
    print(f"\nLargest connected component: {graph.largest_connected_component()}")

    if not args.no_gui:
        visualize.on_circle(graph, title="Graph from graphic sequence")

    if args.randomize > 0:
        graph.randomize(args.randomize)
        print(f"\nAfter {args.randomize} randomizations (adjacency list):")
        _print_adjacency_list(graph)
        print(f"\nLargest connected component: {graph.largest_connected_component()}")
        if not args.no_gui:
            visualize.on_circle(
                graph,
                title=f"Randomized graph ({args.randomize} steps)",
            )


if __name__ == "__main__":
    main()
