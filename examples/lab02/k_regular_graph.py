import argparse

from graphs import Graph, visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a random k-regular graph on n vertices."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=7,
        help="Number of vertices.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Degree of each vertex.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    graph = Graph.create_random_k_regular(args.n, args.k)
    print(f"Generated random {args.k}-regular graph on {args.n} vertices:")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(str(neighbor.v) for neighbor in neighbors)}".rstrip())

    if not args.no_gui:
        visualize.circular(graph, title=f"Random {args.k}-regular graph (n={args.n})")


if __name__ == "__main__":
    main()
