import math
import tkinter as tk
from collections import defaultdict, deque

from graphs.graph import Graph
from graphs.types import AdjacencyList


def _create_canvas(title: str, width: int, height: int) -> tuple[tk.Tk, tk.Canvas]:
    root = tk.Tk()
    root.title(title)
    canvas = tk.Canvas(root, width=width, height=height, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True)
    root.update_idletasks()
    return root, canvas


def _subgraph_edges(subgraph: Graph | None, vertex_count: int) -> set[tuple[int, int]]:
    if subgraph is None:
        return set()
    if subgraph.vertex_count != vertex_count:
        raise ValueError(
            f"Subgraph must have {vertex_count} vertices, got {subgraph.vertex_count}"
        )
    edges: set[tuple[int, int]] = set()
    for u_idx, neighbors in enumerate(subgraph.adjacency_list()):
        for neighbor in neighbors:
            v_idx = neighbor.v - 1
            if u_idx < v_idx:
                edges.add((u_idx, v_idx))
    return edges


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
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return False
    for k, (xk, yk) in enumerate(positions):
        if k in (u_idx, v_idx):
            continue
        t = ((xk - x1) * dx + (yk - y1) * dy) / len_sq
        if 0.0 < t < 1.0:
            px = x1 + t * dx
            py = y1 + t * dy
            if math.hypot(xk - px, yk - py) < threshold:
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
    has_subgraph = subgraph_edges is not None
    highlighted_sub = subgraph_edges or set()

    for u_idx, neighbors in enumerate(adj):
        x1, y1 = positions[u_idx]
        for neighbor in neighbors:
            v_idx = neighbor.v - 1
            if u_idx < v_idx:
                x2, y2 = positions[v_idx]
                in_sub = (u_idx, v_idx) in highlighted_sub
                if in_sub:
                    edge_color, edge_width, text_color = "#2563eb", 4, "#1d4ed8"
                elif has_subgraph:
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
                    side = 1 if (u_idx + v_idx) % 2 == 0 else -1
                    bow = 60.0 * side
                    cx, cy = mx + bow * nx, my + bow * ny
                    canvas.create_line(
                        [x1, y1, cx, cy, x2, y2],
                        smooth=True,
                        fill=edge_color,
                        width=edge_width,
                    )
                    label_x, label_y = mx + 0.5 * bow * nx, my + 0.5 * bow * ny
                else:
                    canvas.create_line(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill=edge_color,
                        width=edge_width,
                    )
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
            x,
            y,
            text=str(i + 1),
            fill="white",
            font=("TkDefaultFont", 11, "bold"),
        )


def _finish_window(root: tk.Tk, canvas: tk.Canvas, show: bool) -> tk.Tk:
    if show:
        root.lift()
        root.update()
        root.after(50, canvas.update)
        root.mainloop()
    return root


def _barycenter(
    vertex: int,
    adj: AdjacencyList,
    pos_in_prev: dict[int, int],
) -> float:
    parents = [neighbor.v for neighbor in adj[vertex - 1] if neighbor.v in pos_in_prev]
    if not parents:
        return 0.0
    return sum(pos_in_prev[p] for p in parents) / len(parents)


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
                v = neighbor.v
                if v not in bfs_dist:
                    bfs_dist[v] = bfs_dist[u] + 1
                    queue.append(v)

    topo_order = sorted(range(1, n + 1), key=lambda v: (bfs_dist[v], v))
    rank = {v: idx for idx, v in enumerate(topo_order)}

    layer: dict[int, int] = {}
    for v in topo_order:
        preds = [nb.v for nb in adj[v - 1] if rank[nb.v] < rank[v]]
        layer[v] = max((layer[p] for p in preds), default=-1) + 1

    columns: dict[int, list[int]] = defaultdict(list)
    for vertex, col in layer.items():
        columns[col].append(vertex)

    for col in columns:
        columns[col].sort()

    max_col = max(columns)
    for _ in range(3):
        for col in range(1, max_col + 1):
            pos_in_prev = {vertex: idx for idx, vertex in enumerate(columns[col - 1])}
            columns[col].sort(key=lambda vertex: _barycenter(vertex, adj, pos_in_prev))

    return dict(columns)


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
    n = graph.vertex_count
    subgraph_edges = _subgraph_edges(subgraph, n) if subgraph is not None else None

    root, canvas = _create_canvas(title, size, size)

    if n > 0:
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
        positions: list[tuple[float, float]] = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = center + r * math.cos(angle)
            y = center + r * math.sin(angle)
            positions.append((x, y))

        _draw_graph(
            canvas,
            graph.adjacency_list(),
            positions,
            vertex_radius,
            show_weights=show_weights,
            subgraph_edges=subgraph_edges,
        )

    return _finish_window(root, canvas, show)


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
    subgraph_edges = _subgraph_edges(subgraph, n) if subgraph is not None else None

    root, canvas = _create_canvas(title, width, height)

    if n > 0:
        adj = graph.adjacency_list()
        columns = _compute_layered_columns(adj, start_vertex=start_vertex)
        margin_x = 70
        margin_y = 60
        num_cols = len(columns)

        positions: list[tuple[float, float]] = [(0.0, 0.0)] * n
        for col_idx, nodes_in_col in columns.items():
            if num_cols == 1:
                x = width / 2
            else:
                col_spacing = (width - 2 * margin_x) / (num_cols - 1)
                x = margin_x + col_idx * col_spacing

            num_nodes = len(nodes_in_col)
            for row_idx, vertex in enumerate(nodes_in_col):
                if num_nodes == 1:
                    y = height / 2
                else:
                    row_spacing = (height - 2 * margin_y) / (num_nodes - 1)
                    y = margin_y + row_idx * row_spacing
                positions[vertex - 1] = (x, y)

        _draw_graph(
            canvas,
            adj,
            positions,
            vertex_radius,
            show_weights=show_weights,
            subgraph_edges=subgraph_edges,
        )

    return _finish_window(root, canvas, show)
