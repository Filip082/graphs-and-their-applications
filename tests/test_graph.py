from pathlib import Path

import numpy as np
import pytest

from graphs import Graph


def test_graph_members_and_conversions() -> None:
    adj_list = [[2, 3], [1], [1]]
    g = Graph(adj_list, "adjacency_list")

    assert g.vertex_count == 3
    assert g.data_type == "adjacency_list"
    assert g.adjacency_list() == adj_list

    expected_adj_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.int64)
    expected_inc_mat = np.array([[1, 1], [1, 0], [0, 1]], dtype=np.int64)

    np.testing.assert_array_equal(g.adjacency_matrix(), expected_adj_mat)
    np.testing.assert_array_equal(g.incidence_matrix(), expected_inc_mat)

    returned_list = g.adjacency_list()
    returned_list[0].append(99)
    assert g.adjacency_list() == adj_list

    g_mat = Graph(expected_adj_mat, "adjacency_matrix")
    assert g_mat.vertex_count == 3
    assert g_mat.data_type == "adjacency_matrix"
    assert g_mat.adjacency_list() == adj_list
    np.testing.assert_array_equal(g_mat.incidence_matrix(), expected_inc_mat)

    g_inc = Graph(expected_inc_mat, "incidence_matrix")
    assert g_inc.vertex_count == 3
    assert g_inc.data_type == "incidence_matrix"
    assert g_inc.adjacency_list() == adj_list
    np.testing.assert_array_equal(g_inc.adjacency_matrix(), expected_adj_mat)


def test_graph_from_file(tmp_path: Path) -> None:
    adj_list_file = tmp_path / "adj_list.txt"
    adj_list_file.write_text("1: 2 3\n2: 1\n3: 1\n4:\n", encoding="utf-8")
    g1 = Graph.from_file(adj_list_file, "adjacency_list")
    assert g1.vertex_count == 4
    assert g1.adjacency_list() == [[2, 3], [1], [1], []]

    adj_mat_file = tmp_path / "adj_mat.txt"
    adj_mat_file.write_text("0 1 0\n1 0 1\n0 1 0\n", encoding="utf-8")
    g2 = Graph.from_file(adj_mat_file, "adjacency_matrix")
    assert g2.vertex_count == 3
    assert g2.adjacency_list() == [[2], [1, 3], [2]]

    inc_mat_file = tmp_path / "inc_mat.txt"
    inc_mat_file.write_text("1 0\n1 1\n0 1\n", encoding="utf-8")
    g3 = Graph.from_file(inc_mat_file, "incidence_matrix")
    assert g3.vertex_count == 3
    assert g3.adjacency_list() == [[2], [1, 3], [2]]


def test_random_binomial() -> None:
    g_empty = Graph.create_random_binomial(5, 0.0)
    assert g_empty.vertex_count == 5
    assert int(g_empty.adjacency_matrix().sum()) == 0

    g_full = Graph.create_random_binomial(5, 1.0)
    assert g_full.vertex_count == 5
    assert int(g_full.adjacency_matrix().sum()) == 20

    g_zero = Graph.create_random_binomial(0, 0.5)
    assert g_zero.vertex_count == 0

    with pytest.raises(ValueError):
        Graph.create_random_binomial(-1, 0.5)
    with pytest.raises(ValueError):
        Graph.create_random_binomial(4, 1.5)


def test_random_uniform() -> None:
    g = Graph.create_random_uniform(6, 7)
    assert g.vertex_count == 6
    assert g.incidence_matrix().shape == (6, 7)
    assert int(g.adjacency_matrix().sum()) == 14

    g_zero = Graph.create_random_uniform(4, 0)
    assert g_zero.incidence_matrix().shape == (4, 0)

    g_full = Graph.create_random_uniform(4, 6)
    assert g_full.incidence_matrix().shape == (4, 6)

    with pytest.raises(ValueError):
        Graph.create_random_uniform(-1, 0)
    with pytest.raises(ValueError):
        Graph.create_random_uniform(4, 7)
