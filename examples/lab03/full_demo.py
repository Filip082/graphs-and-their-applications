import argparse

from graphs import Graph, visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all Lab 03 tasks on a single random connected weighted graph."
    )
    parser.add_argument("--n", type=int, default=12, help="Number of vertices.")
    parser.add_argument(
        "--p", type=float, default=0.35, help="Edge probability for G(n, p)."
    )
    parser.add_argument(
        "--start", "-s", type=int, default=1, help="Start vertex for Dijkstra."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    graph = Graph.create_random_binomial(args.n, args.p, connected=True)

    print("1. Connected random weighted graph (vertex: neighbor(weight)):")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip())

    print(f"\n2. Dijkstra shortest paths (START: s = {args.start}):")
    paths = graph.dijkstra(args.start)
    for vertex in range(1, graph.vertex_count + 1):
        dist, path = paths[vertex]
        print(f"d({vertex}) = {dist} ==> [{' - '.join(str(v) for v in path)}]")

    dist_matrix = graph.distance_matrix()
    max_val = max((int(val) for row in dist_matrix for val in row), default=0)
    width = len(str(max_val))
    print("\n3. Distance matrix:")
    for row in dist_matrix:
        print(" ".join(f"{int(val):>{width}}" for val in row))

    center_v, center_sum = graph.center()
    minimax_v, minimax_dist = graph.minimax()
    print("\n4. Graph centers:")
    print(f"Centrum = {center_v} (suma odleglosci: {center_sum})")
    print(f"Centrum minimax = {minimax_v} (odleglosc od najdalszego: {minimax_dist})")

    mst = graph.kruskal()
    mst_edges = [
        (u, nb.v, nb.w)
        for u, neighbors in enumerate(mst.adjacency_list(), start=1)
        for nb in neighbors
        if u < nb.v
    ]
    total_weight = sum(w for _, _, w in mst_edges)
    print(f"\n5. Minimum spanning tree (total weight = {total_weight}):")
    for u, v, w in mst_edges:
        print(f"  {u} - {v} (w={w})")

    if not args.no_gui:
        visualize.layered(mst, title=f"Lab 03 MST (n={args.n})", show_weights=True)


if __name__ == "__main__":
    main()
