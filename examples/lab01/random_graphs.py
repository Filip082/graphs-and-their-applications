import argparse

from graphs import Graph, visualize


def _print_graph(graph: Graph) -> None:
    print("Adjacency list:")
    for u, neighbors in enumerate(graph.adjacency_list(), start=1):
        print(f"{u}: {' '.join(str(v) for v in neighbors)}".rstrip())
    print("\nAdjacency matrix:")
    print(graph.adjacency_matrix())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random graphs G(n, l) or G(n, p)."
    )
    subparsers = parser.add_subparsers(dest="model", required=True)

    uniform_parser = subparsers.add_parser("uniform", help="Generate G(n, l) graph.")
    uniform_parser.add_argument("--n", type=int, default=7, help="Number of vertices.")
    uniform_parser.add_argument("--l", type=int, default=10, help="Number of edges.")
    uniform_parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )

    binomial_parser = subparsers.add_parser("binomial", help="Generate G(n, p) graph.")
    binomial_parser.add_argument("--n", type=int, default=7, help="Number of vertices.")
    binomial_parser.add_argument(
        "--p", type=float, default=0.5, help="Edge probability."
    )
    binomial_parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip opening the Tkinter visualization window.",
    )

    args = parser.parse_args()

    if args.model == "uniform":
        graph = Graph.create_random_uniform(args.n, args.l)
        title = f"G({args.n}, {args.l})"
    else:
        graph = Graph.create_random_binomial(args.n, args.p)
        title = f"G({args.n}, {args.p})"

    _print_graph(graph)
    if not args.no_gui:
        visualize.circular(graph, title=title)


if __name__ == "__main__":
    main()
