import argparse
import math
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from graphs import Graph, GraphType, visualize


def _create_strongly_connected_weighted_digraph(
    n: int,
    p: float,
    min_weight: int,
    max_weight: int,
    max_attempts: int = 500,
) -> Graph:
    for _ in range(max_attempts):
        graph = Graph.create_random_binomial(
            n, p, directed=True, min_weight=min_weight, max_weight=max_weight
        )
        if n > 0 and len(graph.kosaraju()) != 1:
            continue
        try:
            graph.johnson()
        except ValueError:
            continue
        return graph
    raise RuntimeError(
        "Failed to generate a strongly connected weighted digraph without "
        "negative-weight cycles; try increasing --p or --min-weight."
    )


def _format_distance_matrix(matrix: NDArray[np.float64]) -> str:
    formatted = [
        [
            "inf" if math.isinf(val) else str(int(val) if val.is_integer() else val)
            for val in row
        ]
        for row in matrix
    ]
    width = max((len(cell) for row in formatted for cell in row), default=1)
    return "\n".join(" ".join(f"{cell:>{width}}" for cell in row) for row in formatted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all-pairs shortest paths using Johnson's algorithm on a "
            "strongly connected weighted digraph."
        )
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input weighted digraph file (generates random if omitted).",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=["adjacency_list", "adjacency_matrix", "incidence_matrix"],
        default="adjacency_list",
        help="Representation used in --file.",
    )
    parser.add_argument("--n", type=int, default=7, help="Number of vertices.")
    parser.add_argument(
        "--p",
        type=float,
        default=0.35,
        help="Edge probability for directed G(n, p).",
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=-2,
        help="Minimum random edge weight.",
    )
    parser.add_argument(
        "--max-weight",
        type=int,
        default=10,
        help="Maximum random edge weight.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )
    args = parser.parse_args()

    if args.file is not None:
        graph = Graph.from_file(
            args.file, cast(GraphType, args.data_type), weighted=True
        )
    else:
        graph = _create_strongly_connected_weighted_digraph(
            args.n, args.p, args.min_weight, args.max_weight
        )

    print("Weighted adjacency list (vertex: neighbor(weight)):")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip())

    try:
        dist_matrix = graph.johnson()
    except ValueError as exc:
        print(f"\nJohnson's algorithm failed: {exc}")
        return

    print("\nDistance matrix (Johnson):")
    print(_format_distance_matrix(dist_matrix))

    if not args.no_gui:
        visualize.layered(
            graph,
            title=f"Johnson digraph (n={graph.vertex_count})",
            show_weights=True,
            arrow=True,
        )


if __name__ == "__main__":
    main()
