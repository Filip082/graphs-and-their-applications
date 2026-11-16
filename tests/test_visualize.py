import tkinter as tk

import pytest

from graphs import Graph, Neighbor, visualize
from graphs.visualize import _compute_layered_columns


def _adj(lists: list[list[int]]) -> list[list[Neighbor]]:
    return [[Neighbor(v, 1) for v in neighbors] for neighbors in lists]


def _get_canvas_counts(root: tk.Tk) -> tuple[int, int, int]:
    canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
    assert len(canvases) == 1
    canvas = canvases[0]
    items = canvas.find_all()
    lines = [i for i in items if canvas.type(i) == "line"]
    ovals = [i for i in items if canvas.type(i) == "oval"]
    texts = [i for i in items if canvas.type(i) == "text"]
    return len(lines), len(ovals), len(texts)


def test_circular_draws_vertices_and_edges() -> None:
    g = Graph(_adj([[2, 3], [1], [1], []]), "adjacency_list")
    try:
        root = visualize.circular(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        assert _get_canvas_counts(root) == (2, 5, 4)
    finally:
        root.destroy()


def test_circular_empty_graph() -> None:
    g = Graph([], "adjacency_list")
    try:
        root = visualize.circular(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
        assert len(canvases) == 1
        assert canvases[0].find_all() == ()
    finally:
        root.destroy()


def test_layered_draws_vertices_and_edges() -> None:
    g = Graph(_adj([[2, 3], [1, 4], [1, 4], [2, 3]]), "adjacency_list")
    try:
        root = visualize.layered(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        assert _get_canvas_counts(root) == (4, 4, 4)
    finally:
        root.destroy()


def test_layered_disconnected_and_single_vertex() -> None:
    g = Graph(_adj([[2], [1], []]), "adjacency_list")
    try:
        root = visualize.layered(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        assert _get_canvas_counts(root) == (1, 3, 3)
    finally:
        root.destroy()

    single = Graph([[]], "adjacency_list")
    try:
        root = visualize.layered(single, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        assert _get_canvas_counts(root) == (0, 1, 1)
    finally:
        root.destroy()


def test_layered_empty_graph() -> None:
    g = Graph([], "adjacency_list")
    try:
        root = visualize.layered(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
        assert len(canvases) == 1
        assert canvases[0].find_all() == ()
    finally:
        root.destroy()


def test_layered_invalid_start_vertex() -> None:
    g = Graph(_adj([[2], [1]]), "adjacency_list")
    with pytest.raises(ValueError, match="start_vertex"):
        visualize.layered(g, start_vertex=0, show=False)
    with pytest.raises(ValueError, match="start_vertex"):
        visualize.layered(g, start_vertex=3, show=False)


def test_compute_layered_columns_structure() -> None:
    adj = _adj(
        [
            [2, 3, 5],
            [1, 4, 5],
            [1, 5, 6],
            [2, 7],
            [1, 2, 3, 7, 8],
            [3, 8, 9],
            [4, 5, 10],
            [5, 6, 10, 11, 12],
            [6, 11],
            [7, 8, 12],
            [8, 9, 12],
            [8, 10, 11],
        ]
    )
    columns = _compute_layered_columns(adj, start_vertex=1)
    assert columns[0] == [1]
    placed = [v for col in columns.values() for v in col]
    assert sorted(placed) == list(range(1, 13))
    assert set(columns[1]) == {2, 3, 5}


def test_show_weights_toggle() -> None:
    adj = [
        [Neighbor(2, 5), Neighbor(3, 9)],
        [Neighbor(1, 5)],
        [Neighbor(1, 9)],
    ]
    g = Graph(adj, "adjacency_list")
    try:
        root_circular = visualize.circular(g, show=False, show_weights=True)
        root_layered = visualize.layered(g, show=False, show_weights=True)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        assert _get_canvas_counts(root_circular) == (2, 4, 5)
        assert _get_canvas_counts(root_layered) == (2, 3, 5)
    finally:
        root_circular.destroy()
        root_layered.destroy()
