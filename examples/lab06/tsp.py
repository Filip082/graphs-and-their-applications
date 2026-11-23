import argparse
from pathlib import Path

import numpy as np

from graphs import Graph, visualize


def load_points(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Expected at least 2 coordinates per line, got: {line!r}")
        points.append((float(parts[0]), float(parts[1])))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve 2D TSP using simulated annealing with 2-opt moves."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a 2-column (x y) coordinates file (generates random if omitted).",
    )
    parser.add_argument(
        "--n", type=int, default=40, help="Number of random points if --file omitted."
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=100,
        help="Square board size [0, board_size]^2 for random points.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Iterations per temperature step.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=15,
        help="Number of simulated annealing restarts.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        points = load_points(args.file)
        title_prefix = f"TSP ({args.file.name})"
    else:
        rng = np.random.default_rng(args.seed)
        coords = rng.integers(0, args.board_size + 1, size=(args.n, 2))
        points = [(float(x), float(y)) for x, y in coords]
        title_prefix = f"Random TSP (n={args.n})"

    graph = Graph.from_points(points)
    initial_tour = list(range(1, len(points) + 1))
    initial_len = graph.tour_length(initial_tour)
    best_tour, best_len = graph.simulated_annealing_tsp(
        max_iter=args.max_iter,
        restarts=args.restarts,
        initial_tour=initial_tour,
        seed=args.seed,
    )

    print(f"Vertices: {graph.vertex_count}")
    print(f"Initial tour length:   {initial_len:.3f}")
    print(f"Optimized tour length: {best_len:.3f}")
    print(f"Tour: {best_tour}")

    if not args.no_gui:
        visualize.planar_tour(
            points, best_tour, title=f"{title_prefix} - length: {best_len:.3f}"
        )


if __name__ == "__main__":
    main()
