import math
from pathlib import Path

import numpy as np
import pytest

from graphs import Graph, Neighbor


def _adj(lists: list[list[int]]) -> list[list[Neighbor]]:
    return [[Neighbor(v, 1) for v in neighbors] for neighbors in lists]


def test_graph_members_and_conversions() -> None:
    adj_list = [
        [Neighbor(2, 1), Neighbor(3, 1)],
        [Neighbor(1, 1)],
        [Neighbor(1, 1)],
    ]
    g = Graph(adj_list, "adjacency_list")

    assert g.vertex_count == 3
    assert g.data_type == "adjacency_list"
    assert g.adjacency_list() == adj_list

    expected_adj_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.int64)
    expected_inc_mat = np.array([[1, 1], [1, 0], [0, 1]], dtype=np.int64)

    np.testing.assert_array_equal(g.adjacency_matrix(), expected_adj_mat)
    np.testing.assert_array_equal(g.incidence_matrix(), expected_inc_mat)

    returned_list = g.adjacency_list()
    returned_list[0].append(Neighbor(99, 1))
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
    assert g1.adjacency_list() == _adj([[2, 3], [1], [1], []])

    weighted_adj_list_file = tmp_path / "weighted_adj_list.txt"
    weighted_adj_list_file.write_text(
        "1: 2 5 3 -1\n2: 1 5\n3: 1 -1\n4:\n",
        encoding="utf-8",
    )
    g1_w = Graph.from_file(weighted_adj_list_file, "adjacency_list", weighted=True)
    assert g1_w.vertex_count == 4
    assert g1_w.adjacency_list() == [
        [Neighbor(2, 5), Neighbor(3, -1)],
        [Neighbor(1, 5)],
        [Neighbor(1, -1)],
        [],
    ]

    bad_weighted_file = tmp_path / "bad_weighted.txt"
    bad_weighted_file.write_text("1: 2 5 3\n2: 1 5\n3: 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Graph.from_file(bad_weighted_file, "adjacency_list", weighted=True)

    adj_mat_file = tmp_path / "adj_mat.txt"
    adj_mat_file.write_text("0 1 0\n1 0 1\n0 1 0\n", encoding="utf-8")
    g2 = Graph.from_file(adj_mat_file, "adjacency_matrix")
    assert g2.vertex_count == 3
    assert g2.adjacency_list() == _adj([[2], [1, 3], [2]])

    inc_mat_file = tmp_path / "inc_mat.txt"
    inc_mat_file.write_text("1 0\n1 1\n0 1\n", encoding="utf-8")
    g3 = Graph.from_file(inc_mat_file, "incidence_matrix")
    assert g3.vertex_count == 3
    assert g3.adjacency_list() == _adj([[2], [1, 3], [2]])


def _assert_weights_in_range(graph: Graph, low: int = 1, high: int = 10) -> None:
    for neighbors in graph.adjacency_list():
        for neighbor in neighbors:
            assert low <= neighbor.w <= high


def test_random_binomial() -> None:
    g_empty = Graph.create_random_binomial(5, 0.0)
    assert g_empty.vertex_count == 5
    assert int(g_empty.adjacency_matrix().sum()) == 0

    g_full = Graph.create_random_binomial(5, 1.0)
    assert g_full.vertex_count == 5
    full_matrix = g_full.adjacency_matrix()
    assert int(np.count_nonzero(full_matrix)) == 20
    _assert_weights_in_range(g_full)

    g_zero = Graph.create_random_binomial(0, 0.5)
    assert g_zero.vertex_count == 0

    with pytest.raises(ValueError):
        Graph.create_random_binomial(-1, 0.5)
    with pytest.raises(ValueError):
        Graph.create_random_binomial(4, 1.5)
    with pytest.raises(ValueError):
        Graph.create_random_binomial(3, 0.0, connected=True)

    g_conn = Graph.create_random_binomial(10, 0.01, connected=True)
    assert len(g_conn.connected_components()) == 1
    _assert_weights_in_range(g_conn)
    assert (
        len(Graph.create_random_binomial(1, 0.0, connected=True).connected_components())
        == 1
    )


def test_random_uniform() -> None:
    g = Graph.create_random_uniform(6, 7)
    assert g.vertex_count == 6
    assert g.incidence_matrix().shape == (6, 7)
    assert int(np.count_nonzero(g.adjacency_matrix())) == 14
    _assert_weights_in_range(g)

    g_zero = Graph.create_random_uniform(4, 0)
    assert g_zero.incidence_matrix().shape == (4, 0)

    g_full = Graph.create_random_uniform(4, 6)
    assert g_full.incidence_matrix().shape == (4, 6)

    with pytest.raises(ValueError):
        Graph.create_random_uniform(-1, 0)
    with pytest.raises(ValueError):
        Graph.create_random_uniform(4, 7)
    with pytest.raises(ValueError):
        Graph.create_random_uniform(5, 3, connected=True)

    for edges in (7, 10):
        g_conn = Graph.create_random_uniform(8, edges, connected=True)
        assert g_conn.incidence_matrix().shape == (8, edges)
        assert len(g_conn.connected_components()) == 1
        _assert_weights_in_range(g_conn)


def test_from_sequence() -> None:
    g_empty = Graph.from_sequence([])
    assert g_empty.vertex_count == 0

    seq = [4, 3, 3, 2, 2, 2]
    g = Graph.from_sequence(seq)
    assert g.vertex_count == len(seq)
    assert [len(neighbors) for neighbors in g.adjacency_list()] == seq

    with pytest.raises(ValueError):
        Graph.from_sequence([3, 3, 1, 1])


def test_randomize_preserves_degrees_and_representation() -> None:
    for data_type in ("adjacency_list", "adjacency_matrix", "incidence_matrix"):
        base = Graph.from_sequence([3, 3, 3, 3, 3, 3])
        if data_type == "adjacency_matrix":
            g = Graph(base.adjacency_matrix(), "adjacency_matrix")
        elif data_type == "incidence_matrix":
            g = Graph(base.incidence_matrix(), "incidence_matrix")
        else:
            g = base

        degrees_before = [len(n) for n in g.adjacency_list()]
        g.randomize(0)
        assert [len(n) for n in g.adjacency_list()] == degrees_before

        g.randomize(20)
        assert g.data_type == data_type
        assert [len(n) for n in g.adjacency_list()] == degrees_before

    k4 = Graph.from_sequence([3, 3, 3, 3])
    k4.randomize(10)
    assert [len(n) for n in k4.adjacency_list()] == [3, 3, 3, 3]

    with pytest.raises(ValueError):
        k4.randomize(-1)


def test_create_random_k_regular() -> None:
    g0 = Graph.create_random_k_regular(0, 0)
    assert g0.vertex_count == 0

    for n, k in [(4, 0), (4, 3), (6, 2), (6, 3), (8, 4)]:
        g = Graph.create_random_k_regular(n, k)
        assert g.vertex_count == n
        assert all(len(neighbors) == k for neighbors in g.adjacency_list())
        _assert_weights_in_range(g)

    for n, k in [(-1, 0), (4, -1), (0, 1), (4, 4), (5, 3)]:
        with pytest.raises(ValueError):
            Graph.create_random_k_regular(n, k)

    for n, k in [(2, 1), (6, 2), (8, 3), (10, 2)]:
        g_conn = Graph.create_random_k_regular(n, k, connected=True)
        assert g_conn.vertex_count == n
        assert all(len(neighbors) == k for neighbors in g_conn.adjacency_list())
        assert len(g_conn.connected_components()) == 1
        _assert_weights_in_range(g_conn)

    for n, k in [(2, 0), (4, 1)]:
        with pytest.raises(ValueError):
            Graph.create_random_k_regular(n, k, connected=True)


def _verify_euler_cycle(graph: Graph, cycle: list[int]) -> None:
    adj = graph.adjacency_list()
    expected_edges = {
        (min(u, neighbor.v), max(u, neighbor.v))
        for u, neighbors in enumerate(adj, start=1)
        for neighbor in neighbors
        if u < neighbor.v
    }
    assert cycle[0] == cycle[-1]
    assert len(cycle) == len(expected_edges) + 1
    traversed: list[tuple[int, int]] = []
    for u, v in zip(cycle[:-1], cycle[1:], strict=True):
        assert any(neighbor.v == v for neighbor in adj[u - 1])
        traversed.append((min(u, v), max(u, v)))
    assert len(traversed) == len(set(traversed))
    assert set(traversed) == expected_edges


def test_create_random_eulerian_and_find_euler_cycle() -> None:
    assert Graph.create_random_eulerian(0).find_euler_cycle() == []
    assert Graph.create_random_eulerian(1).find_euler_cycle() == [1]

    for n in (3, 4, 5, 6, 8, 10):
        g = Graph.create_random_eulerian(n, connected=True)
        assert g.vertex_count == n
        assert all(len(neighbors) % 2 == 0 for neighbors in g.adjacency_list())
        assert len(g.largest_connected_component()) == n
        _assert_weights_in_range(g)
        cycle = g.find_euler_cycle()
        assert cycle is not None
        _verify_euler_cycle(g, cycle)

    bowtie = Graph(
        _adj([[2, 3, 4, 5], [1, 3], [1, 2], [1, 5], [1, 4]]),
        "adjacency_list",
    )
    bowtie_cycle = bowtie.find_euler_cycle()
    assert bowtie_cycle is not None
    _verify_euler_cycle(bowtie, bowtie_cycle)

    odd_graph = Graph(_adj([[2], [1, 3], [2]]), "adjacency_list")
    assert odd_graph.find_euler_cycle() is None

    disconnected_even = Graph(
        _adj([[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5]]),
        "adjacency_list",
    )
    assert disconnected_even.find_euler_cycle() is None

    with pytest.raises(ValueError):
        Graph.create_random_eulerian(-1)
    with pytest.raises(ValueError):
        Graph.create_random_eulerian(2)


def test_largest_connected_component() -> None:
    assert Graph([], "adjacency_list").connected_components() == []
    assert Graph([], "adjacency_list").largest_connected_component() == []
    assert Graph([[]], "adjacency_list").connected_components() == [[1]]
    assert Graph([[]], "adjacency_list").largest_connected_component() == [1]

    g = Graph(
        _adj([[2], [1], [4, 5], [3, 5], [3, 4], []]),
        "adjacency_list",
    )
    assert g.connected_components() == [[1, 2], [3, 4, 5], [6]]
    assert g.largest_connected_component() == [3, 4, 5]


def test_is_hamiltonian() -> None:
    assert Graph([], "adjacency_list").is_hamiltonian() is False
    assert Graph([[]], "adjacency_list").is_hamiltonian() is False
    assert Graph(_adj([[2], [1]]), "adjacency_list").is_hamiltonian() is False

    c4 = Graph(_adj([[2, 4], [1, 3], [2, 4], [1, 3]]), "adjacency_list")
    assert c4.is_hamiltonian() is True

    k5 = Graph.create_random_k_regular(5, 4)
    assert k5.is_hamiltonian() is True

    bowtie = Graph(
        _adj([[2, 3, 4, 5], [1, 3], [1, 2], [1, 5], [1, 4]]),
        "adjacency_list",
    )
    assert bowtie.is_hamiltonian() is False

    petersen = Graph(
        _adj(
            [
                [2, 5, 6],
                [1, 3, 7],
                [2, 4, 8],
                [3, 5, 9],
                [1, 4, 10],
                [1, 8, 9],
                [2, 9, 10],
                [3, 6, 10],
                [4, 6, 7],
                [5, 7, 8],
            ]
        ),
        "adjacency_list",
    )
    assert petersen.is_hamiltonian() is False


def _verify_hamiltonian_cycle(graph: Graph, cycle: list[int]) -> None:
    n = graph.vertex_count
    assert len(cycle) == n + 1
    assert cycle[0] == cycle[-1]
    assert set(cycle[:-1]) == set(range(1, n + 1))
    adj = graph.adjacency_list()
    for u, v in zip(cycle, cycle[1:], strict=False):
        assert any(nb.v == v for nb in adj[u - 1])


def test_find_hamiltonian_cycle() -> None:
    assert Graph([], "adjacency_list").find_hamiltonian_cycle() is None
    assert Graph([[]], "adjacency_list").find_hamiltonian_cycle() is None
    assert Graph(_adj([[2], [1]]), "adjacency_list").find_hamiltonian_cycle() is None

    c4 = Graph(_adj([[2, 4], [1, 3], [2, 4], [1, 3]]), "adjacency_list")
    cycle_c4 = c4.find_hamiltonian_cycle()
    assert cycle_c4 is not None
    _verify_hamiltonian_cycle(c4, cycle_c4)

    k5 = Graph.create_random_k_regular(5, 4)
    cycle_k5 = k5.find_hamiltonian_cycle()
    assert cycle_k5 is not None
    _verify_hamiltonian_cycle(k5, cycle_k5)

    bowtie = Graph(
        _adj([[2, 3, 4, 5], [1, 3], [1, 2], [1, 5], [1, 4]]),
        "adjacency_list",
    )
    assert bowtie.find_hamiltonian_cycle() is None


def test_weighted_graph_storage_and_unweighted_algorithms() -> None:
    adj_list = [
        [Neighbor(2, 3), Neighbor(3, 5)],
        [Neighbor(1, 3), Neighbor(3, 2)],
        [Neighbor(1, 5), Neighbor(2, 2)],
    ]
    g_list = Graph(adj_list, "adjacency_list")
    expected_mat = np.array([[0, 3, 5], [3, 0, 2], [5, 2, 0]], dtype=np.int64)
    expected_inc = np.array([[3, 5, 0], [3, 0, 2], [0, 5, 2]], dtype=np.int64)

    np.testing.assert_array_equal(g_list.adjacency_matrix(), expected_mat)
    np.testing.assert_array_equal(g_list.incidence_matrix(), expected_inc)

    g_mat = Graph(expected_mat, "adjacency_matrix")
    assert g_mat.adjacency_list() == adj_list
    np.testing.assert_array_equal(g_mat.incidence_matrix(), expected_inc)

    g_inc = Graph(expected_inc, "incidence_matrix")
    assert g_inc.adjacency_list() == adj_list
    np.testing.assert_array_equal(g_inc.adjacency_matrix(), expected_mat)

    assert g_list.connected_components() == [[1, 2, 3]]
    assert g_list.is_hamiltonian() is True
    cycle = g_list.find_euler_cycle()
    assert cycle is not None
    _verify_euler_cycle(g_list, cycle)


def test_dijkstra_task3_example() -> None:
    adj_list = [
        [Neighbor(2, 3), Neighbor(3, 2), Neighbor(5, 9)],
        [Neighbor(1, 3), Neighbor(4, 2), Neighbor(5, 4)],
        [Neighbor(1, 2), Neighbor(5, 6), Neighbor(6, 9)],
        [Neighbor(2, 2), Neighbor(7, 3)],
        [
            Neighbor(1, 9),
            Neighbor(2, 4),
            Neighbor(3, 6),
            Neighbor(7, 1),
            Neighbor(8, 2),
        ],
        [Neighbor(3, 9), Neighbor(8, 1), Neighbor(9, 2)],
        [Neighbor(4, 3), Neighbor(5, 1), Neighbor(10, 5)],
        [
            Neighbor(5, 2),
            Neighbor(6, 1),
            Neighbor(10, 5),
            Neighbor(11, 6),
            Neighbor(12, 9),
        ],
        [Neighbor(6, 2), Neighbor(11, 2)],
        [Neighbor(7, 5), Neighbor(8, 5), Neighbor(12, 5)],
        [Neighbor(8, 6), Neighbor(9, 2), Neighbor(12, 3)],
        [Neighbor(8, 9), Neighbor(10, 5), Neighbor(11, 3)],
    ]
    expected = {
        1: (0, [1]),
        2: (3, [1, 2]),
        3: (2, [1, 3]),
        4: (5, [1, 2, 4]),
        5: (7, [1, 2, 5]),
        6: (10, [1, 2, 5, 8, 6]),
        7: (8, [1, 2, 4, 7]),
        8: (9, [1, 2, 5, 8]),
        9: (12, [1, 2, 5, 8, 6, 9]),
        10: (13, [1, 2, 4, 7, 10]),
        11: (14, [1, 2, 5, 8, 6, 9, 11]),
        12: (17, [1, 2, 5, 8, 6, 9, 11, 12]),
    }

    g_list = Graph(adj_list, "adjacency_list")
    assert g_list.dijkstra(1) == expected

    g_mat = Graph(g_list.adjacency_matrix(), "adjacency_matrix")
    assert g_mat.dijkstra(1) == expected

    g_inc = Graph(g_list.incidence_matrix(), "incidence_matrix")
    assert g_inc.dijkstra(1) == expected


def test_dijkstra_edge_cases() -> None:
    single = Graph([[]], "adjacency_list")
    assert single.dijkstra(1) == {1: (0, [1])}

    disconnected = Graph(
        [
            [Neighbor(2, 4)],
            [Neighbor(1, 4)],
            [Neighbor(4, 2)],
            [Neighbor(3, 2)],
        ],
        "adjacency_list",
    )
    assert disconnected.dijkstra(1) == {
        1: (0, [1]),
        2: (4, [1, 2]),
        3: (math.inf, []),
        4: (math.inf, []),
    }

    empty = Graph([], "adjacency_list")
    with pytest.raises(ValueError):
        empty.dijkstra(1)

    for bad in (0, -1, 5, True):
        with pytest.raises(ValueError):
            disconnected.dijkstra(bad)

    neg = Graph([[Neighbor(2, -1)], [Neighbor(1, -1)]], "adjacency_list")
    with pytest.raises(ValueError):
        neg.dijkstra(1)


def test_distance_matrix_task3_example() -> None:
    adj_list = [
        [Neighbor(2, 3), Neighbor(3, 2), Neighbor(5, 9)],
        [Neighbor(1, 3), Neighbor(4, 2), Neighbor(5, 4)],
        [Neighbor(1, 2), Neighbor(5, 6), Neighbor(6, 9)],
        [Neighbor(2, 2), Neighbor(7, 3)],
        [
            Neighbor(1, 9),
            Neighbor(2, 4),
            Neighbor(3, 6),
            Neighbor(7, 1),
            Neighbor(8, 2),
        ],
        [Neighbor(3, 9), Neighbor(8, 1), Neighbor(9, 2)],
        [Neighbor(4, 3), Neighbor(5, 1), Neighbor(10, 5)],
        [
            Neighbor(5, 2),
            Neighbor(6, 1),
            Neighbor(10, 5),
            Neighbor(11, 6),
            Neighbor(12, 9),
        ],
        [Neighbor(6, 2), Neighbor(11, 2)],
        [Neighbor(7, 5), Neighbor(8, 5), Neighbor(12, 5)],
        [Neighbor(8, 6), Neighbor(9, 2), Neighbor(12, 3)],
        [Neighbor(8, 9), Neighbor(10, 5), Neighbor(11, 3)],
    ]
    expected = np.array(
        [
            [0, 3, 2, 5, 7, 10, 8, 9, 12, 13, 14, 17],
            [3, 0, 5, 2, 4, 7, 5, 6, 9, 10, 11, 14],
            [2, 5, 0, 7, 6, 9, 7, 8, 11, 12, 13, 16],
            [5, 2, 7, 0, 4, 7, 3, 6, 9, 8, 11, 13],
            [7, 4, 6, 4, 0, 3, 1, 2, 5, 6, 7, 10],
            [10, 7, 9, 7, 3, 0, 4, 1, 2, 6, 4, 7],
            [8, 5, 7, 3, 1, 4, 0, 3, 6, 5, 8, 10],
            [9, 6, 8, 6, 2, 1, 3, 0, 3, 5, 5, 8],
            [12, 9, 11, 9, 5, 2, 6, 3, 0, 8, 2, 5],
            [13, 10, 12, 8, 6, 6, 5, 5, 8, 0, 8, 5],
            [14, 11, 13, 11, 7, 4, 8, 5, 2, 8, 0, 3],
            [17, 14, 16, 13, 10, 7, 10, 8, 5, 5, 3, 0],
        ],
        dtype=np.int64,
    )

    g = Graph(adj_list, "adjacency_list")
    np.testing.assert_array_equal(g.distance_matrix(), expected)


def test_distance_matrix_edge_cases() -> None:
    empty = Graph([], "adjacency_list")
    np.testing.assert_array_equal(
        empty.distance_matrix(), np.zeros((0, 0), dtype=np.int64)
    )

    single = Graph([[]], "adjacency_list")
    np.testing.assert_array_equal(
        single.distance_matrix(), np.zeros((1, 1), dtype=np.int64)
    )

    disconnected = Graph(
        [
            [Neighbor(2, 4)],
            [Neighbor(1, 4)],
            [Neighbor(4, 2)],
            [Neighbor(3, 2)],
        ],
        "adjacency_list",
    )
    with pytest.raises(ValueError):
        disconnected.distance_matrix()


def test_center_and_minimax_task3_example() -> None:
    adj_list = [
        [Neighbor(2, 3), Neighbor(3, 2), Neighbor(5, 9)],
        [Neighbor(1, 3), Neighbor(4, 2), Neighbor(5, 4)],
        [Neighbor(1, 2), Neighbor(5, 6), Neighbor(6, 9)],
        [Neighbor(2, 2), Neighbor(7, 3)],
        [
            Neighbor(1, 9),
            Neighbor(2, 4),
            Neighbor(3, 6),
            Neighbor(7, 1),
            Neighbor(8, 2),
        ],
        [Neighbor(3, 9), Neighbor(8, 1), Neighbor(9, 2)],
        [Neighbor(4, 3), Neighbor(5, 1), Neighbor(10, 5)],
        [
            Neighbor(5, 2),
            Neighbor(6, 1),
            Neighbor(10, 5),
            Neighbor(11, 6),
            Neighbor(12, 9),
        ],
        [Neighbor(6, 2), Neighbor(11, 2)],
        [Neighbor(7, 5), Neighbor(8, 5), Neighbor(12, 5)],
        [Neighbor(8, 6), Neighbor(9, 2), Neighbor(12, 3)],
        [Neighbor(8, 9), Neighbor(10, 5), Neighbor(11, 3)],
    ]
    g = Graph(adj_list, "adjacency_list")
    assert g.center() == (5, 55)
    assert g.minimax() == (8, 9)


def test_center_and_minimax_edge_cases() -> None:
    single = Graph([[]], "adjacency_list")
    assert single.center() == (1, 0)
    assert single.minimax() == (1, 0)

    empty = Graph([], "adjacency_list")
    with pytest.raises(ValueError):
        empty.center()
    with pytest.raises(ValueError):
        empty.minimax()

    disconnected = Graph(
        [
            [Neighbor(2, 4)],
            [Neighbor(1, 4)],
            [Neighbor(4, 2)],
            [Neighbor(3, 2)],
        ],
        "adjacency_list",
    )
    with pytest.raises(ValueError):
        disconnected.center()
    with pytest.raises(ValueError):
        disconnected.minimax()


def _mst_edge_count_and_weight(tree: Graph) -> tuple[int, int]:
    edges = [
        (u_idx, neighbor.v - 1, neighbor.w)
        for u_idx, neighbors in enumerate(tree.adjacency_list())
        for neighbor in neighbors
        if u_idx < neighbor.v - 1
    ]
    return len(edges), sum(w for _, _, w in edges)


def test_prim_and_kruskal_task3_example() -> None:
    adj_list = [
        [Neighbor(2, 3), Neighbor(3, 2), Neighbor(5, 9)],
        [Neighbor(1, 3), Neighbor(4, 2), Neighbor(5, 4)],
        [Neighbor(1, 2), Neighbor(5, 6), Neighbor(6, 9)],
        [Neighbor(2, 2), Neighbor(7, 3)],
        [
            Neighbor(1, 9),
            Neighbor(2, 4),
            Neighbor(3, 6),
            Neighbor(7, 1),
            Neighbor(8, 2),
        ],
        [Neighbor(3, 9), Neighbor(8, 1), Neighbor(9, 2)],
        [Neighbor(4, 3), Neighbor(5, 1), Neighbor(10, 5)],
        [
            Neighbor(5, 2),
            Neighbor(6, 1),
            Neighbor(10, 5),
            Neighbor(11, 6),
            Neighbor(12, 9),
        ],
        [Neighbor(6, 2), Neighbor(11, 2)],
        [Neighbor(7, 5), Neighbor(8, 5), Neighbor(12, 5)],
        [Neighbor(8, 6), Neighbor(9, 2), Neighbor(12, 3)],
        [Neighbor(8, 9), Neighbor(10, 5), Neighbor(11, 3)],
    ]
    g = Graph(adj_list, "adjacency_list")
    orig_matrix = g.adjacency_matrix()

    for tree in (g.prim(), g.kruskal()):
        assert tree.vertex_count == 12
        assert len(tree.connected_components()) == 1
        edge_count, total_weight = _mst_edge_count_and_weight(tree)
        assert edge_count == 11
        assert total_weight == 26
        for u_idx, neighbors in enumerate(tree.adjacency_list()):
            for neighbor in neighbors:
                assert orig_matrix[u_idx, neighbor.v - 1] == neighbor.w


def test_prim_and_kruskal_edge_cases() -> None:
    empty = Graph([], "adjacency_list")
    assert empty.prim().vertex_count == 0
    assert empty.kruskal().vertex_count == 0

    single = Graph([[]], "adjacency_list")
    assert single.prim().adjacency_list() == [[]]
    assert single.kruskal().adjacency_list() == [[]]

    disconnected = Graph(
        [
            [Neighbor(2, 4)],
            [Neighbor(1, 4)],
            [Neighbor(4, 2)],
            [Neighbor(3, 2)],
        ],
        "adjacency_list",
    )
    with pytest.raises(ValueError):
        disconnected.prim()
    with pytest.raises(ValueError):
        disconnected.kruskal()


def test_digraph_with_negative_weights_and_from_file(tmp_path: Path) -> None:
    digraph_file = tmp_path / "digraph.txt"
    digraph_file.write_text("1: 2 5 3 -2\n2: 3 1\n3: 1 4\n", encoding="utf-8")

    g = Graph.from_file(digraph_file, "adjacency_list", weighted=True)
    expected_adj = [
        [Neighbor(2, 5), Neighbor(3, -2)],
        [Neighbor(3, 1)],
        [Neighbor(1, 4)],
    ]
    expected_mat = np.array(
        [
            [0, 5, -2],
            [0, 0, 1],
            [4, 0, 0],
        ],
        dtype=np.int64,
    )
    assert g.adjacency_list() == expected_adj
    np.testing.assert_array_equal(g.adjacency_matrix(), expected_mat)

    g_mat = Graph(expected_mat, "adjacency_matrix")
    assert g_mat.adjacency_list() == expected_adj

    g_inc = g.incidence_matrix()
    assert Graph(g_inc, "directed_incidence_matrix").adjacency_list() == expected_adj
    inc_file = tmp_path / "directed_inc.txt"
    inc_file.write_text(
        "\n".join(" ".join(str(x) for x in row) for row in g_inc) + "\n",
        encoding="utf-8",
    )
    assert (
        Graph.from_file(inc_file, "directed_incidence_matrix").adjacency_list()
        == expected_adj
    )
    with pytest.raises(ValueError):
        g.dijkstra(1)

    for method in (
        lambda: g.randomize(1),
        g.connected_components,
        g.largest_connected_component,
        g.find_euler_cycle,
        g.find_hamiltonian_cycle,
        g.is_hamiltonian,
        g.prim,
        g.kruskal,
    ):
        with pytest.raises(ValueError):
            method()


def test_digraph_dijkstra_positive_weights() -> None:
    g = Graph(
        [
            [Neighbor(2, 2), Neighbor(3, 10)],
            [Neighbor(3, 3)],
            [],
        ],
        "adjacency_list",
    )
    assert g.dijkstra(1) == {
        1: (0, [1]),
        2: (2, [1, 2]),
        3: (5, [1, 2, 3]),
    }
    assert g.dijkstra(3) == {
        1: (math.inf, []),
        2: (math.inf, []),
        3: (0, [3]),
    }


def _is_weakly_connected(graph: Graph) -> bool:
    mat = graph.adjacency_matrix()
    sym = ((mat != 0) | (mat.T != 0)).astype(np.int64)
    return len(Graph(sym, "adjacency_matrix").connected_components()) == 1


def test_random_generators_directed() -> None:
    g_full = Graph.create_random_binomial(4, 1.0, directed=True)
    assert int(np.count_nonzero(g_full.adjacency_matrix())) == 12
    _assert_weights_in_range(g_full)

    g_conn = Graph.create_random_binomial(8, 0.01, connected=True, directed=True)
    assert _is_weakly_connected(g_conn)
    _assert_weights_in_range(g_conn)

    g_uni_full = Graph.create_random_uniform(4, 12, directed=True)
    assert int(np.count_nonzero(g_uni_full.adjacency_matrix())) == 12
    _assert_weights_in_range(g_uni_full)

    with pytest.raises(ValueError):
        Graph.create_random_uniform(4, 13, directed=True)

    g_uni_conn = Graph.create_random_uniform(6, 5, connected=True, directed=True)
    assert int(np.count_nonzero(g_uni_conn.adjacency_matrix())) == 5
    assert _is_weakly_connected(g_uni_conn)
    _assert_weights_in_range(g_uni_conn)


def test_random_generators_weight_bounds() -> None:
    g_neg = Graph.create_random_binomial(
        8, 1.0, directed=True, min_weight=-5, max_weight=5
    )
    assert int(np.count_nonzero(g_neg.adjacency_matrix())) == 56
    _assert_weights_in_range(g_neg, low=-5, high=5)

    g_exact = Graph.create_random_uniform(
        5, 6, directed=True, min_weight=-3, max_weight=-3
    )
    _assert_weights_in_range(g_exact, low=-3, high=-3)

    g_reg = Graph.create_random_k_regular(6, 2, min_weight=-4, max_weight=-1)
    _assert_weights_in_range(g_reg, low=-4, high=-1)

    g_eul = Graph.create_random_eulerian(5, min_weight=2, max_weight=2)
    _assert_weights_in_range(g_eul, low=2, high=2)

    for bad_min, bad_max in [(5, 2), (0, 0), (True, 5), (1, False)]:
        with pytest.raises(ValueError):
            Graph.create_random_binomial(4, 0.5, min_weight=bad_min, max_weight=bad_max)
        with pytest.raises(ValueError):
            Graph.create_random_uniform(4, 3, min_weight=bad_min, max_weight=bad_max)
        with pytest.raises(ValueError):
            Graph.create_random_k_regular(4, 2, min_weight=bad_min, max_weight=bad_max)
        with pytest.raises(ValueError):
            Graph.create_random_eulerian(4, min_weight=bad_min, max_weight=bad_max)


def _task4_example_graph() -> Graph:
    return Graph(
        [
            [Neighbor(2, 6), Neighbor(3, 3), Neighbor(5, -1)],
            [
                Neighbor(1, 10),
                Neighbor(3, -5),
                Neighbor(4, -4),
                Neighbor(5, 4),
                Neighbor(7, 4),
            ],
            [Neighbor(6, 2)],
            [Neighbor(2, 5), Neighbor(7, 9)],
            [Neighbor(7, -4)],
            [Neighbor(2, 9)],
            [Neighbor(6, 4)],
        ],
        "adjacency_list",
    )


def test_kosaraju() -> None:
    g_task4 = _task4_example_graph()
    assert g_task4.kosaraju() == [[1, 2, 3, 4, 5, 6, 7]]
    assert Graph(g_task4.adjacency_matrix(), "adjacency_matrix").kosaraju() == [
        [1, 2, 3, 4, 5, 6, 7]
    ]

    g_multi = Graph(
        [
            [Neighbor(2, 1)],
            [Neighbor(3, 1)],
            [Neighbor(1, 1), Neighbor(4, 1)],
            [Neighbor(5, 1)],
            [Neighbor(4, 1)],
            [],
        ],
        "adjacency_list",
    )
    sccs = {tuple(comp) for comp in g_multi.kosaraju()}
    assert sccs == {(1, 2, 3), (4, 5), (6,)}

    assert Graph([], "adjacency_list").kosaraju() == []
    assert Graph([[]], "adjacency_list").kosaraju() == [[1]]


def test_bellman_ford() -> None:
    g_task4 = _task4_example_graph()
    res = g_task4.bellman_ford(1)
    assert res[1] == (0, [1])
    assert res[2] == (6, [1, 2])
    assert res[3] == (1, [1, 2, 3])
    assert res[4] == (2, [1, 2, 4])
    assert res[5] == (-1, [1, 5])
    assert res[6] == (-1, [1, 5, 7, 6])
    assert res[7] == (-5, [1, 5, 7])

    g_disconnected = Graph(
        [[Neighbor(2, -3)], [], [Neighbor(4, 2)], []],
        "adjacency_list",
    )
    assert g_disconnected.bellman_ford(1) == {
        1: (0, [1]),
        2: (-3, [1, 2]),
        3: (math.inf, []),
        4: (math.inf, []),
    }

    g_neg_cycle = Graph(
        [
            [Neighbor(2, 1)],
            [Neighbor(3, -2)],
            [Neighbor(1, -1)],
            [Neighbor(1, 5)],
        ],
        "adjacency_list",
    )
    with pytest.raises(ValueError):
        g_neg_cycle.bellman_ford(1)

    g_unreachable_neg = Graph(
        [
            [Neighbor(2, 3)],
            [],
            [Neighbor(4, -2)],
            [Neighbor(3, -1)],
        ],
        "adjacency_list",
    )
    assert g_unreachable_neg.bellman_ford(1) == {
        1: (0, [1]),
        2: (3, [1, 2]),
        3: (math.inf, []),
        4: (math.inf, []),
    }

    for bad in (0, -1, 5, True):
        with pytest.raises(ValueError):
            g_disconnected.bellman_ford(bad)


def test_johnson() -> None:
    g_task4 = _task4_example_graph()
    expected = np.array(
        [
            [0, 6, 1, 2, -1, -1, -5],
            [10, 0, -5, -4, 4, -3, 0],
            [21, 11, 0, 7, 15, 2, 11],
            [15, 5, 0, 0, 9, 2, 5],
            [19, 9, 4, 5, 0, 0, -4],
            [19, 9, 4, 5, 13, 0, 9],
            [23, 13, 8, 9, 17, 4, 0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(g_task4.johnson(), expected)
    np.testing.assert_array_equal(
        Graph(g_task4.adjacency_matrix(), "adjacency_matrix").johnson(), expected
    )

    g_dag = Graph(
        [[Neighbor(2, 4), Neighbor(3, 1)], [Neighbor(3, -5)], []],
        "adjacency_list",
    )
    expected_dag = np.array(
        [
            [0, 4, -1],
            [math.inf, 0, -5],
            [math.inf, math.inf, 0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(g_dag.johnson(), expected_dag)

    g_neg_cycle = Graph(
        [[Neighbor(2, 2)], [Neighbor(1, -3)]],
        "adjacency_list",
    )
    with pytest.raises(ValueError):
        g_neg_cycle.johnson()

    assert Graph([], "adjacency_list").johnson().shape == (0, 0)
    np.testing.assert_array_equal(
        Graph([[]], "adjacency_list").johnson(), np.array([[0.0]])
    )


def test_create_random_flow_network() -> None:
    for n in (2, 3, 4):
        g = Graph.create_random_flow_network(n)
        mat = g.adjacency_matrix()
        v_count = g.vertex_count
        assert 2 + 2 * n <= v_count <= 2 + n * n
        assert np.all(mat[:, 0] == 0)
        assert np.all(mat[v_count - 1, :] == 0)
        assert np.all(np.diag(mat) == 0)
        _assert_weights_in_range(g, low=1, high=10)
        assert np.any(mat[0, :] > 0)
        assert np.any(mat[:, v_count - 1] > 0)

    for bad_n in (0, 1, -2, True):
        with pytest.raises(ValueError):
            Graph.create_random_flow_network(bad_n)
    with pytest.raises(ValueError):
        Graph.create_random_flow_network(2, min_weight=0, max_weight=5)


def test_ford_fulkerson() -> None:
    # Classic CLRS flow network (6 vertices: s=1, t=6, max flow=23)
    clrs = Graph(
        [
            [Neighbor(2, 16), Neighbor(3, 13)],
            [Neighbor(3, 10), Neighbor(4, 12)],
            [Neighbor(2, 4), Neighbor(5, 14)],
            [Neighbor(3, 9), Neighbor(6, 20)],
            [Neighbor(4, 7), Neighbor(6, 4)],
            [],
        ],
        "adjacency_list",
    )
    max_flow, flow_graph = clrs.ford_fulkerson(1, 6)
    assert max_flow == 23

    cap = clrs.adjacency_matrix()
    flow = flow_graph.adjacency_matrix()
    assert np.all(flow >= 0)
    assert np.all(flow <= cap)
    assert int(flow[0, :].sum() - flow[:, 0].sum()) == 23
    assert int(flow[:, 5].sum() - flow[5, :].sum()) == 23
    for v in range(1, 5):
        assert int(flow[:, v].sum()) == int(flow[v, :].sum())

    # Disconnected sink -> max flow is 0
    disconnected = Graph(
        [[Neighbor(2, 5)], [], [Neighbor(4, 3)], []],
        "adjacency_list",
    )
    zero_flow, zero_graph = disconnected.ford_fulkerson(1, 4)
    assert zero_flow == 0
    assert np.all(zero_graph.adjacency_matrix() == 0)

    # Random flow network conservation and capacity constraints
    net = Graph.create_random_flow_network(3)
    net_max, net_flow_graph = net.ford_fulkerson(1, net.vertex_count)
    net_cap = net.adjacency_matrix()
    net_flow = net_flow_graph.adjacency_matrix()
    assert net_max > 0
    assert np.all(net_flow >= 0)
    assert np.all(net_flow <= net_cap)
    assert int(net_flow[0, :].sum() - net_flow[:, 0].sum()) == net_max

    with pytest.raises(ValueError):
        clrs.ford_fulkerson(1, 1)
    with pytest.raises(ValueError):
        clrs.ford_fulkerson(0, 6)
    with pytest.raises(ValueError):
        clrs.ford_fulkerson(1, 7)

    neg_cap = Graph([[Neighbor(2, -1)], []], "adjacency_list")
    with pytest.raises(ValueError):
        neg_cap.ford_fulkerson(1, 2)


def test_ford_fulkerson_input5_example() -> None:
    from pathlib import Path

    data_path = (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "lab05"
        / "data"
        / "flow_network_adj_list.txt"
    )
    g = Graph.from_file(data_path, "adjacency_list", weighted=True)
    max_flow, flow_graph = g.ford_fulkerson(1, 11)
    assert max_flow == 19
    flow = flow_graph.adjacency_matrix()
    assert int(flow[0, 1]) == 10
    assert int(flow[0, 2]) == 3
    assert int(flow[0, 3]) == 6
    assert int(flow[1, 2]) == 5
    assert int(flow[1, 4]) == 0
    assert int(flow[1, 5]) == 5
    assert int(flow[2, 5]) == 2
    assert int(flow[2, 6]) == 6
    assert int(flow[3, 4]) == 5
    assert int(flow[3, 6]) == 1
    assert int(flow[4, 8]) == 5
    assert int(flow[5, 9]) == 7
    assert int(flow[6, 7]) == 7
    assert int(flow[7, 10]) == 7
    assert int(flow[8, 10]) == 5
    assert int(flow[9, 10]) == 7
