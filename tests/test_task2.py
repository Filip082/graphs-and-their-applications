from typing import Any, cast

import numpy as np
import pytest

from graphs import Graph, utils


def test_is_graphic_sequence() -> None:
    assert utils.is_graphic_sequence([]) is True
    assert utils.is_graphic_sequence([0, 0, 0]) is True
    assert utils.is_graphic_sequence([3, 3, 3, 3]) is True
    assert utils.is_graphic_sequence([4, 2, 2, 2, 2]) is True
    assert utils.is_graphic_sequence([4, 3, 3, 2, 2, 2]) is True
    assert utils.is_graphic_sequence(np.array([2, 2, 2], dtype=np.int64)) is True

    assert utils.is_graphic_sequence([3, 2, 2]) is False
    assert utils.is_graphic_sequence([4, 2, 2, 2]) is False
    assert utils.is_graphic_sequence([2, 2, -2]) is False
    assert utils.is_graphic_sequence([3, 3, 1, 1]) is False
    assert utils.is_graphic_sequence([4, 4, 4, 2, 0]) is False
    assert utils.is_graphic_sequence([5, 5, 3, 3, 2, 2]) is True

    original = [3, 2, 2, 1]
    assert utils.is_graphic_sequence(original) is True
    assert original == [3, 2, 2, 1]

    with pytest.raises(ValueError):
        utils.is_graphic_sequence(cast(Any, [True, 1]))
    with pytest.raises(ValueError):
        utils.is_graphic_sequence(cast(Any, [1.5, 1.5]))


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

    for n, k in [(-1, 0), (4, -1), (0, 1), (4, 4), (5, 3)]:
        with pytest.raises(ValueError):
            Graph.create_random_k_regular(n, k)


def _verify_euler_cycle(graph: Graph, cycle: list[int]) -> None:
    adj = graph.adjacency_list()
    expected_edges = {
        (min(u, v), max(u, v))
        for u, neighbors in enumerate(adj)
        for v in neighbors
        if u < v
    }
    assert cycle[0] == cycle[-1]
    assert len(cycle) == len(expected_edges) + 1
    traversed: list[tuple[int, int]] = []
    for u, v in zip(cycle[:-1], cycle[1:], strict=True):
        assert v in adj[u]
        traversed.append((min(u, v), max(u, v)))
    assert len(traversed) == len(set(traversed))
    assert set(traversed) == expected_edges


def test_create_random_eulerian_and_find_euler_cycle() -> None:
    assert Graph.create_random_eulerian(0).find_euler_cycle() == []
    assert Graph.create_random_eulerian(1).find_euler_cycle() == [0]

    for n in (3, 4, 5, 6, 8, 10):
        g = Graph.create_random_eulerian(n)
        assert g.vertex_count == n
        assert all(len(neighbors) % 2 == 0 for neighbors in g.adjacency_list())
        assert len(g.largest_connected_component()) == n
        cycle = g.find_euler_cycle()
        assert cycle is not None
        _verify_euler_cycle(g, cycle)

    bowtie = Graph(
        [[1, 2, 3, 4], [0, 2], [0, 1], [0, 4], [0, 3]],
        "adjacency_list",
    )
    bowtie_cycle = bowtie.find_euler_cycle()
    assert bowtie_cycle is not None
    _verify_euler_cycle(bowtie, bowtie_cycle)

    odd_graph = Graph([[1], [0, 2], [1]], "adjacency_list")
    assert odd_graph.find_euler_cycle() is None

    disconnected_even = Graph(
        [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]],
        "adjacency_list",
    )
    assert disconnected_even.find_euler_cycle() is None

    with pytest.raises(ValueError):
        Graph.create_random_eulerian(-1)
    with pytest.raises(ValueError):
        Graph.create_random_eulerian(2)


def test_largest_connected_component() -> None:
    assert Graph([], "adjacency_list").largest_connected_component() == []
    assert Graph([[]], "adjacency_list").largest_connected_component() == [0]

    g = Graph(
        [[1], [0], [3, 4], [2, 4], [2, 3], []],
        "adjacency_list",
    )
    assert g.largest_connected_component() == [2, 3, 4]


def test_is_hamiltonian() -> None:
    assert Graph([], "adjacency_list").is_hamiltonian() is False
    assert Graph([[]], "adjacency_list").is_hamiltonian() is False
    assert Graph([[1], [0]], "adjacency_list").is_hamiltonian() is False

    c4 = Graph([[1, 3], [0, 2], [1, 3], [0, 2]], "adjacency_list")
    assert c4.is_hamiltonian() is True

    k5 = Graph.create_random_k_regular(5, 4)
    assert k5.is_hamiltonian() is True

    bowtie = Graph(
        [[1, 2, 3, 4], [0, 2], [0, 1], [0, 4], [0, 3]],
        "adjacency_list",
    )
    assert bowtie.is_hamiltonian() is False

    petersen = Graph(
        [
            [1, 4, 5],
            [0, 2, 6],
            [1, 3, 7],
            [2, 4, 8],
            [0, 3, 9],
            [0, 7, 8],
            [1, 8, 9],
            [2, 5, 9],
            [3, 5, 6],
            [4, 6, 7],
        ],
        "adjacency_list",
    )
    assert petersen.is_hamiltonian() is False
