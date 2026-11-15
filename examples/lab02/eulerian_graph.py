import argparse

from graphs import Graph, visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a random Eulerian graph and find an Euler cycle."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Number of vertices.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    graph = Graph.create_random_eulerian(args.n)
    adj = graph.adjacency_list()
    print(f"Generated random Eulerian graph on {args.n} vertices:")
    print(f"Degrees: {[len(neighbors) for neighbors in adj]}")
    for u, neighbors in enumerate(adj, start=1):
        print(f"{u}: {' '.join(str(v) for v in neighbors)}".rstrip())

    cycle = graph.find_euler_cycle()
    if cycle is None:
        print("\nNo Euler cycle found.")
    else:
        print(f"\nEuler cycle: {' - '.join(str(v) for v in cycle)}")

    if not args.no_gui:
        visualize.on_circle(graph, title=f"Random Eulerian graph (n={args.n})")


if __name__ == "__main__":
    main()
