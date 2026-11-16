import math
import tkinter as tk
from collections import defaultdict, deque
from collections.abc import Callable

from graphs.graph import Graph
from graphs.types import AdjacencyList


def _subgraph_edges(
    subgraph: Graph | None, vertex_count: int
) -> set[tuple[int, int]] | None:
    if subgraph is None:
        return None
    if subgraph.vertex_count != vertex_count:
        raise ValueError(
            f"Subgraph must have {vertex_count} vertices, got {subgraph.vertex_count}"
        )
    return {
        (u_idx, neighbor.v - 1)
        for u_idx, neighbors in enumerate(subgraph.adjacency_list())
        for neighbor in neighbors
        if u_idx < neighbor.v - 1
    }


def _edge_collides_with_vertex(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    positions: list[tuple[float, float]],
    u_idx: int,
    v_idx: int,
    threshold: float,
) -> bool:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return False
    for k, (xk, yk) in enumerate(positions):
        if k in (u_idx, v_idx):
            continue
        t = ((xk - x1) * dx + (yk - y1) * dy) / len_sq
        if (
            0.0 < t < 1.0
            and math.hypot(xk - (x1 + t * dx), yk - (y1 + t * dy)) < threshold
        ):
            return True
    return False


def _draw_graph(
    canvas: tk.Canvas,
    adj: AdjacencyList,
    positions: list[tuple[float, float]],
    vertex_radius: int,
    show_weights: bool = False,
    subgraph_edges: set[tuple[int, int]] | None = None,
) -> None:
    highlighted = subgraph_edges or set()

    for u_idx, neighbors in enumerate(adj):
        x1, y1 = positions[u_idx]
        for neighbor in neighbors:
            v_idx = neighbor.v - 1
            if u_idx >= v_idx:
                continue
            x2, y2 = positions[v_idx]
            if (u_idx, v_idx) in highlighted:
                edge_color, edge_width, text_color = "#2563eb", 4, "#1d4ed8"
            elif subgraph_edges is not None:
                edge_color, edge_width, text_color = "#cbd5e1", 1.5, "#64748b"
            else:
                edge_color, edge_width, text_color = "#334155", 2, "#0f172a"

            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if _edge_collides_with_vertex(
                x1, y1, x2, y2, positions, u_idx, v_idx, vertex_radius + 4
            ):
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                nx, ny = -dy / length, dx / length
                bow = 60.0 * (1 if (u_idx + v_idx) % 2 == 0 else -1)
                canvas.create_line(
                    [x1, y1, mx + bow * nx, my + bow * ny, x2, y2],
                    smooth=True,
                    fill=edge_color,
                    width=edge_width,
                )
                label_x, label_y = mx + 0.5 * bow * nx, my + 0.5 * bow * ny
            else:
                canvas.create_line(x1, y1, x2, y2, fill=edge_color, width=edge_width)
                label_x, label_y = mx, my

            if show_weights:
                label = str(neighbor.w)
                pad_x = max(8, 4 * len(label) + 4)
                canvas.create_rectangle(
                    label_x - pad_x,
                    label_y - 8,
                    label_x + pad_x,
                    label_y + 8,
                    fill="white",
                    outline="",
                )
                canvas.create_text(
                    label_x,
                    label_y,
                    text=label,
                    fill=text_color,
                    font=("TkDefaultFont", 10, "bold"),
                )

    for i, (x, y) in enumerate(positions):
        canvas.create_oval(
            x - vertex_radius,
            y - vertex_radius,
            x + vertex_radius,
            y + vertex_radius,
            fill="#3b82f6",
            outline="#1e3a8a",
            width=2,
        )
        canvas.create_text(
            x, y, text=str(i + 1), fill="white", font=("TkDefaultFont", 11, "bold")
        )


def _compute_layered_columns(
    adj: AdjacencyList,
    start_vertex: int = 1,
) -> dict[int, list[int]]:
    n = len(adj)
    if n == 0:
        return {}

    bfs_dist: dict[int, int] = {}
    roots = [start_vertex] + [v for v in range(1, n + 1) if v != start_vertex]
    for root_vertex in roots:
        if root_vertex in bfs_dist:
            continue
        bfs_dist[root_vertex] = 0
        queue = deque([root_vertex])
        while queue:
            u = queue.popleft()
            for neighbor in adj[u - 1]:
                if neighbor.v not in bfs_dist:
                    bfs_dist[neighbor.v] = bfs_dist[u] + 1
                    queue.append(neighbor.v)

    topo_order = sorted(range(1, n + 1), key=lambda v: (bfs_dist[v], v))
    rank = {v: idx for idx, v in enumerate(topo_order)}
    layer: dict[int, int] = {}
    for v in topo_order:
        preds = [nb.v for nb in adj[v - 1] if rank[nb.v] < rank[v]]
        layer[v] = max((layer[p] for p in preds), default=-1) + 1

    columns: dict[int, list[int]] = defaultdict(list)
    for vertex, col in layer.items():
        columns[col].append(vertex)
    for nodes in columns.values():
        nodes.sort()

    def barycenter(vertex: int, pos_in_prev: dict[int, int]) -> float:
        parents = [nb.v for nb in adj[vertex - 1] if nb.v in pos_in_prev]
        return sum(pos_in_prev[p] for p in parents) / len(parents) if parents else 0.0

    max_col = max(columns)
    for _ in range(3):
        for col in range(1, max_col + 1):
            pos_in_prev = {v: idx for idx, v in enumerate(columns[col - 1])}
            columns[col].sort(key=lambda v: barycenter(v, pos_in_prev))

    return dict(columns)


def _render_graph(
    graph: Graph,
    width: int,
    height: int,
    vertex_radius: int,
    title: str,
    show: bool,
    show_weights: bool,
    subgraph: Graph | None,
    layout: Callable[[tk.Canvas, AdjacencyList], list[tuple[float, float]]],
) -> tk.Tk:
    sub_edges = _subgraph_edges(subgraph, graph.vertex_count)
    root = tk.Tk()
    root.title(title)
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True)
    root.update_idletasks()

    if graph.vertex_count > 0:
        adj = graph.adjacency_list()
        positions = layout(canvas, adj)
        _draw_graph(canvas, adj, positions, vertex_radius, show_weights, sub_edges)

    if show:
        root.lift()
        root.update()
        root.after(50, canvas.update)
        root.mainloop()
    return root


def circular(
    graph: Graph,
    *,
    size: int = 600,
    radius: float | None = None,
    vertex_radius: int = 16,
    title: str = "Graph",
    show: bool = True,
    show_weights: bool = False,
    subgraph: Graph | None = None,
) -> tk.Tk:
    def layout(canvas: tk.Canvas, adj: AdjacencyList) -> list[tuple[float, float]]:
        n = len(adj)
        center = size / 2
        r = radius if radius is not None else size * 0.36
        canvas.create_oval(
            center - r,
            center - r,
            center + r,
            center + r,
            outline="red",
            width=1,
            dash=(4, 4),
        )
        return [
            (
                center + r * math.cos(2 * math.pi * i / n - math.pi / 2),
                center + r * math.sin(2 * math.pi * i / n - math.pi / 2),
            )
            for i in range(n)
        ]

    return _render_graph(
        graph, size, size, vertex_radius, title, show, show_weights, subgraph, layout
    )


def layered(
    graph: Graph,
    *,
    width: int = 900,
    height: int = 380,
    vertex_radius: int = 16,
    start_vertex: int = 1,
    title: str = "Graph",
    show: bool = True,
    show_weights: bool = False,
    subgraph: Graph | None = None,
) -> tk.Tk:
    n = graph.vertex_count
    if n > 0 and not 1 <= start_vertex <= n:
        raise ValueError(f"start_vertex must be between 1 and {n}, got {start_vertex}")

    def layout(_canvas: tk.Canvas, adj: AdjacencyList) -> list[tuple[float, float]]:
        columns = _compute_layered_columns(adj, start_vertex=start_vertex)
        margin_x, margin_y = 70, 60
        num_cols = len(columns)
        positions = [(0.0, 0.0)] * n
        for col_idx, nodes in columns.items():
            x = (
                width / 2
                if num_cols == 1
                else margin_x + col_idx * (width - 2 * margin_x) / (num_cols - 1)
            )
            num_nodes = len(nodes)
            for row_idx, vertex in enumerate(nodes):
                y = (
                    height / 2
                    if num_nodes == 1
                    else margin_y + row_idx * (height - 2 * margin_y) / (num_nodes - 1)
                )
                positions[vertex - 1] = (x, y)
        return positions

    return _render_graph(
        graph, width, height, vertex_radius, title, show, show_weights, subgraph, layout
    )
