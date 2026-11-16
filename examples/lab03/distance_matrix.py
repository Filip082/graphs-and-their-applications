import argparse

from graphs import Graph, visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the all-pairs shortest-path distance matrix."
    )
    parser.add_argument("--n", type=int, default=12, help="Number of vertices.")
    parser.add_argument(
        "--p", type=float, default=0.35, help="Edge probability for G(n, p)."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    graph = Graph.create_random_binomial(args.n, args.p, connected=True)

    print("Weighted adjacency list (vertex: neighbor(weight)):")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip())

    dist_matrix = graph.distance_matrix()
    max_val = max((int(val) for row in dist_matrix for val in row), default=0)
    width = len(str(max_val))
    print("\nDistance matrix:")
    for row in dist_matrix:
        print(" ".join(f"{int(val):>{width}}" for val in row))

    if not args.no_gui:
        visualize.layered(graph, title=f"Distance matrix graph (n={args.n})")


if __name__ == "__main__":
    main()
