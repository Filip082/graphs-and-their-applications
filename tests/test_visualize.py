import tkinter as tk

import pytest

from graphs import Graph, visualize


def test_on_circle_draws_vertices_and_edges() -> None:
    g = Graph([[2, 3], [1], [1], []], "adjacency_list")
    try:
        root = visualize.on_circle(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
        assert len(canvases) == 1
        canvas = canvases[0]
        items = canvas.find_all()
        lines = [i for i in items if canvas.type(i) == "line"]
        ovals = [i for i in items if canvas.type(i) == "oval"]
        texts = [i for i in items if canvas.type(i) == "text"]
        assert len(lines) == 2
        assert len(ovals) == 5
        assert len(texts) == 4
    finally:
        root.destroy()


def test_on_circle_empty_graph() -> None:
    g = Graph([], "adjacency_list")
    try:
        root = visualize.on_circle(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
        assert len(canvases) == 1
        assert canvases[0].find_all() == ()
    finally:
        root.destroy()
