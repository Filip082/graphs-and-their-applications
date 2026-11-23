import argparse
from pathlib import Path
from typing import cast

import numpy as np

from graphs import Graph, GraphType, visualize


def _ensure_outgoing_edges(graph: Graph, seed: int | None = None) -> Graph:
    mat = graph.adjacency_matrix().copy()
    n = graph.vertex_count
    if n <= 1:
        return graph
    rng = np.random.default_rng(seed)
    for u in range(n):
        if np.all(mat[u] == 0):
            v = int(rng.integers(0, n - 1))
            if v >= u:
                v += 1
            mat[u, v] = 1
    return Graph(mat, "adjacency_matrix")


def _print_ranking(title: str, scores: np.ndarray) -> None:
    print(title)
    order = np.argsort(-scores, kind="stable")
    for idx in order:
        print(f"  {int(idx) + 1:>2} ==> PageRank = {scores[idx]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PageRank for a digraph using random walk and power iteration."
        )
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input digraph file (generates random if omitted).",
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
    parser.add_argument("--n", type=int, default=12, help="Number of vertices.")
    parser.add_argument(
        "--p", type=float, default=0.25, help="Edge probability for random digraph."
    )
    parser.add_argument(
        "--d", type=float, default=0.15, help="Teleportation probability d."
    )
    parser.add_argument(
        "--steps", type=int, default=1_000_000, help="Random walk steps."
    )
    parser.add_argument(
        "--max-iter", type=int, default=10_000, help="Max power-iteration steps."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        graph = Graph.from_file(args.file, cast(GraphType, args.data_type))
        title = f"Digraph ({args.file.name})"
    else:
        raw = Graph.create_random_binomial(
            args.n, args.p, directed=True, min_weight=1, max_weight=1
        )
        graph = _ensure_outgoing_edges(raw, seed=args.seed)
        title = f"Random digraph G({args.n}, {args.p})"

    rw = graph.pagerank_random_walk(d=args.d, steps=args.steps, seed=args.seed)
    pi = graph.pagerank_power_iteration(d=args.d, max_iter=args.max_iter)

    _print_ranking(f"Random walk (d={args.d}, steps={args.steps}):", rw)
    print()
    _print_ranking(f"Power iteration (d={args.d}):", pi)
    if len(rw) > 0:
        print(f"\nMax absolute difference: {float(np.max(np.abs(rw - pi))):.6f}")

    if not args.no_gui:
        visualize.layered(graph, title=title, arrow=True)


if __name__ == "__main__":
    main()
