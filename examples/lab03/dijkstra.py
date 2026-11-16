import argparse

from graphs import Graph, visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Dijkstra's shortest-path algorithm on a random weighted graph."
    )
    parser.add_argument("--n", type=int, default=12, help="Number of vertices.")
    parser.add_argument(
        "--p", type=float, default=0.35, help="Edge probability for G(n, p)."
    )
    parser.add_argument(
        "--start", "-s", type=int, default=1, help="Start vertex (1-based)."
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

    paths = graph.dijkstra(args.start)
    print(f"\nSTART: s = {args.start}")
    for vertex in range(1, graph.vertex_count + 1):
        dist, path = paths[vertex]
        formatted_path = " - ".join(str(v) for v in path)
        print(f"d({vertex}) = {dist} ==> [{formatted_path}]")

    if not args.no_gui:
        visualize.layered(
            graph, start_vertex=args.start, title=f"Dijkstra from s={args.start}"
        )


if __name__ == "__main__":
    main()
