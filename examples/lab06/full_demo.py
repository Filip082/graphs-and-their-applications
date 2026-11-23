import argparse
from pathlib import Path

import numpy as np

from examples.lab06.pagerank import _ensure_outgoing_edges, _print_ranking
from examples.lab06.tsp import load_points
from graphs import Graph, visualize


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Lab 06 tasks.")
    parser.add_argument(
        "--digraph-file",
        type=Path,
        default=Path(__file__).parent / "data" / "pagerank_adj_list.txt",
        help="Path to digraph adjacency list file.",
    )
    parser.add_argument(
        "--tsp-file",
        type=Path,
        default=Path(__file__).parent / "data" / "tsp_points.dat",
        help="Path to 2D TSP coordinates file.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use randomly generated inputs instead of data files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization windows.",
    )
    args = parser.parse_args()

    if args.random:
        digraph = _ensure_outgoing_edges(
            Graph.create_random_binomial(
                12, 0.25, directed=True, min_weight=1, max_weight=1
            ),
            seed=args.seed,
        )
        rng = np.random.default_rng(args.seed)
        points = [(float(x), float(y)) for x, y in rng.integers(0, 101, size=(36, 2))]
    else:
        digraph = Graph.from_file(args.digraph_file, "adjacency_list")
        points = load_points(args.tsp_file)

    print("=== Task 1: PageRank ===")
    rw = digraph.pagerank_random_walk(d=0.15, steps=1_000_000, seed=args.seed)
    pi = digraph.pagerank_power_iteration(d=0.15)
    _print_ranking("Random walk (d=0.15, steps=1000000):", rw)
    print()
    _print_ranking("Power iteration (d=0.15):", pi)
    print(f"Max absolute difference: {float(np.max(np.abs(rw - pi))):.6f}")

    print("\n=== Task 2: TSP (Simulated Annealing + 2-opt) ===")
    tsp_graph = Graph.from_points(points)
    initial_len = tsp_graph.tour_length(list(range(1, len(points) + 1)))
    best_tour, best_len = tsp_graph.simulated_annealing_tsp(
        max_iter=2000, restarts=2, seed=args.seed
    )
    print(f"Initial tour length:   {initial_len:.3f}")
    print(f"Optimized tour length: {best_len:.3f}")

    if not args.no_gui:
        visualize.layered(digraph, title="Lab 06 - PageRank Digraph", arrow=True)
        visualize.planar_tour(
            points, best_tour, title=f"Lab 06 - TSP Tour (length={best_len:.3f})"
        )


if __name__ == "__main__":
    main()
