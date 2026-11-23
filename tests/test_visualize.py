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
    assert [len(columns[i]) for i in range(6)] == [1, 2, 3, 3, 2, 1]
    assert set(columns[1]) == {2, 3}


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


def test_subgraph_highlighting() -> None:
    g = Graph(_adj([[2, 3], [1, 3], [1, 2]]), "adjacency_list")
    mst = Graph(_adj([[2, 3], [1], [1]]), "adjacency_list")
    try:
        root = visualize.layered(g, show=False, subgraph=mst)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
        canvas = canvases[0]
        blue_lines = [
            i
            for i in canvas.find_all()
            if canvas.type(i) == "line" and canvas.itemcget(i, "fill") == "#2563eb"
        ]
        faded_lines = [
            i
            for i in canvas.find_all()
            if canvas.type(i) == "line" and canvas.itemcget(i, "fill") == "#cbd5e1"
        ]
        assert len(blue_lines) == 2
        assert len(faded_lines) == 1
    finally:
        root.destroy()


def test_invalid_subgraph_vertex_count() -> None:
    g = Graph(_adj([[2], [1]]), "adjacency_list")
    other = Graph(_adj([[2], [1], []]), "adjacency_list")
    with pytest.raises(ValueError, match="Subgraph must have 2 vertices"):
        visualize.circular(g, show=False, subgraph=other)
    with pytest.raises(ValueError, match="Subgraph must have 2 vertices"):
        visualize.layered(g, show=False, subgraph=other)


def test_layered_curves_collinear_edges() -> None:
    g = Graph(_adj([[2, 3], [1, 3], [1, 2]]), "adjacency_list")
    try:
        root = visualize.layered(g, show=False)
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
        canvas = canvases[0]
        curved = [
            i
            for i in canvas.find_all()
            if canvas.type(i) == "line" and len(canvas.coords(i)) > 4
        ]
        assert len(curved) == 1
    finally:
        root.destroy()


def test_arrow_option() -> None:
    g = Graph(
        [
            [Neighbor(2, 3)],
            [Neighbor(1, -1)],
            [Neighbor(1, 2)],
        ],
        "adjacency_list",
    )
    sub = Graph([[Neighbor(2, 3)], [], []], "adjacency_list")
    try:
        root_no_arrow = visualize.circular(g, show=False)
        root_arrow = visualize.circular(
            g, show=False, show_weights=True, subgraph=sub, arrow=True
        )
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        assert _get_canvas_counts(root_no_arrow) == (2, 4, 3)
        canvases_no = [
            w for w in root_no_arrow.winfo_children() if isinstance(w, tk.Canvas)
        ]
        lines_no = [
            i for i in canvases_no[0].find_all() if canvases_no[0].type(i) == "line"
        ]
        assert all(canvases_no[0].itemcget(i, "arrow") == "none" for i in lines_no)

        assert _get_canvas_counts(root_arrow) == (3, 4, 6)
        canvases_yes = [
            w for w in root_arrow.winfo_children() if isinstance(w, tk.Canvas)
        ]
        canvas = canvases_yes[0]
        lines_yes = [i for i in canvas.find_all() if canvas.type(i) == "line"]
        assert all(canvas.itemcget(i, "arrow") == "last" for i in lines_yes)
        curved = [i for i in lines_yes if len(canvas.coords(i)) > 4]
        assert len(curved) == 2
        highlighted = [i for i in lines_yes if canvas.itemcget(i, "fill") == "#2563eb"]
        assert len(highlighted) == 1
    finally:
        root_no_arrow.destroy()
        root_arrow.destroy()


def test_edge_labels_option() -> None:
    g = Graph([[Neighbor(2, 10)], []], "adjacency_list")
    try:
        root = visualize.layered(
            g,
            show=False,
            show_weights=True,
            arrow=True,
            edge_labels={(1, 2): "7/10"},
        )
    except tk.TclError:
        pytest.skip("Tk display not available")

    try:
        canvases = [w for w in root.winfo_children() if isinstance(w, tk.Canvas)]
        canvas = canvases[0]
        texts = [
            canvas.itemcget(i, "text")
            for i in canvas.find_all()
            if canvas.type(i) == "text"
        ]
        assert "7/10" in texts
    finally:
        root.destroy()
