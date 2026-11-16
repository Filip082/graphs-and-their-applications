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


def _draw_graph(
    canvas: tk.Canvas,
    adj: AdjacencyList,
    positions: list[tuple[float, float]],
    vertex_radius: int,
    show_weights: bool = False,
) -> None:
    for u_idx, neighbors in enumerate(adj):
        x1, y1 = positions[u_idx]
        for neighbor in neighbors:
            v_idx = neighbor.v - 1
            if u_idx < v_idx:
                x2, y2 = positions[v_idx]
                canvas.create_line(x1, y1, x2, y2, fill="#334155", width=2)
                if show_weights:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    label = str(neighbor.w)
                    pad_x = max(8, 4 * len(label) + 4)
                    canvas.create_rectangle(
                        mx - pad_x,
                        my - 8,
                        mx + pad_x,
                        my + 8,
                        fill="white",
                        outline="",
                    )
                    canvas.create_text(
                        mx,
                        my,
                        text=label,
                        fill="#0f172a",
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

    layer: dict[int, int] = {}
    roots = [start_vertex] + [v for v in range(1, n + 1) if v != start_vertex]
    for root_vertex in roots:
        if root_vertex in layer:
            continue
        layer[root_vertex] = 0
        queue = deque([root_vertex])
        while queue:
            u = queue.popleft()
            for neighbor in adj[u - 1]:
                v = neighbor.v
                if v not in layer:
                    layer[v] = layer[u] + 1
                    queue.append(v)

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
) -> tk.Tk:
    root, canvas = _create_canvas(title, size, size)

    n = graph.vertex_count
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
) -> tk.Tk:
    n = graph.vertex_count
    if n > 0 and not 1 <= start_vertex <= n:
        raise ValueError(f"start_vertex must be between 1 and {n}, got {start_vertex}")

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
        )

    return _finish_window(root, canvas, show)
