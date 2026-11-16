import argparse

from graphs import Graph, visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find center and minimax center of a random weighted graph."
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

    dist_matrix = graph.distance_matrix()
    max_val = max((int(val) for row in dist_matrix for val in row), default=0)
    width = len(str(max_val))
    print("Distance matrix:")
    for row in dist_matrix:
        print(" ".join(f"{int(val):>{width}}" for val in row))

    center_v, center_sum = graph.center()
    minimax_v, minimax_dist = graph.minimax()
    print(f"\nCentrum = {center_v} (suma odleglosci: {center_sum})")
    print(f"Centrum minimax = {minimax_v} (odleglosc od najdalszego: {minimax_dist})")

    if not args.no_gui:
        visualize.layered(graph, title=f"Graph center (n={args.n})", show_weights=True)


if __name__ == "__main__":
    main()
