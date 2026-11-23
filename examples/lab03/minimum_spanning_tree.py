import argparse
from pathlib import Path
from typing import cast

from graphs import Graph, GraphType, visualize


def _mst_edges(mst: Graph) -> list[tuple[int, int, int | float]]:
    edges: list[tuple[int, int, int | float]] = []
    for u, neighbors in enumerate(mst.adjacency_list(), start=1):
        for nb in neighbors:
            if u < nb.v:
                edges.append((u, nb.v, nb.w))
    return edges


def _print_mst(name: str, mst: Graph) -> None:
    edges = _mst_edges(mst)
    total_weight = sum(w for _, _, w in edges)
    print(f"\n{name} MST (total weight = {total_weight}):")
    for u, v, w in edges:
        print(f"  {u} - {v} (w={w})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a minimum spanning tree using Prim's or Kruskal's algorithm."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to an input weighted graph file (generates random if omitted).",
    )
    parser.add_argument(
        "--type",
        dest="data_type",
        choices=["adjacency_list", "adjacency_matrix", "incidence_matrix"],
        default="adjacency_list",
        help="Representation used in --file.",
    )
    parser.add_argument("--n", type=int, default=12, help="Number of vertices.")
    parser.add_argument(
        "--p", type=float, default=0.35, help="Edge probability for G(n, p)."
    )
    parser.add_argument(
        "--algorithm",
        choices=["prim", "kruskal", "both"],
        default="both",
        help="MST algorithm to run.",
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
        graph = Graph.create_random_binomial(args.n, args.p, connected=True)

    print("Weighted adjacency list (vertex: neighbor(weight)):")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(f'{nb.v}({nb.w})' for nb in neighbors)}".rstrip())

    mst_to_show: Graph | None = None
    if args.algorithm in ("prim", "both"):
        prim_mst = graph.prim()
        _print_mst("Prim", prim_mst)
        mst_to_show = prim_mst

    if args.algorithm in ("kruskal", "both"):
        kruskal_mst = graph.kruskal()
        _print_mst("Kruskal", kruskal_mst)
        if mst_to_show is None:
            mst_to_show = kruskal_mst

    if not args.no_gui and mst_to_show is not None:
        visualize.layered(
            mst_to_show,
            title=f"Minimum Spanning Tree (n={graph.vertex_count})",
            show_weights=True,
        )


if __name__ == "__main__":
    main()
