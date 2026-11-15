import math
import tkinter as tk

from graphs.graph import Graph


def on_circle(
    graph: Graph,
    *,
    size: int = 600,
    radius: float | None = None,
    vertex_radius: int = 16,
    title: str = "Graph",
    show: bool = True,
) -> tk.Tk:
    root = tk.Tk()
    root.title(title)

    canvas = tk.Canvas(root, width=size, height=size, bg="white")
    canvas.pack(fill=tk.BOTH, expand=True)
    root.update_idletasks()

    n = graph.vertex_count
    if n > 0:
        center = size / 2
        r = radius if radius is not None else size * 0.36
        positions: list[tuple[float, float]] = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = center + r * math.cos(angle)
            y = center + r * math.sin(angle)
            positions.append((x, y))

        adj = graph.adjacency_list()
        for u, neighbors in enumerate(adj):
            x1, y1 = positions[u]
            for v in neighbors:
                if u < v:
                    x2, y2 = positions[v]
                    canvas.create_line(x1, y1, x2, y2, fill="#334155", width=2)

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
                text=str(i),
                fill="white",
                font=("TkDefaultFont", 11, "bold"),
            )

    if show:
        root.lift()
        root.update()
        root.after(50, canvas.update)
        root.mainloop()

    return root
