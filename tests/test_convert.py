from typing import Any

import numpy as np
import pytest

from graphs import Neighbor, convert
from graphs.types import AdjacencyList


@pytest.fixture
def sample_adj_list() -> AdjacencyList:
    return [
        [Neighbor(2, 1), Neighbor(3, 1)],
        [Neighbor(1, 1), Neighbor(3, 1)],
        [Neighbor(1, 1), Neighbor(2, 1), Neighbor(4, 1)],
        [Neighbor(3, 1)],
    ]


@pytest.fixture
def sample_adj_matrix() -> np.ndarray:
    return np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.int64,
    )


@pytest.fixture
def sample_inc_matrix() -> np.ndarray:
    return np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.int64,
    )


def test_all_six_conversions(
    sample_adj_list: AdjacencyList,
    sample_adj_matrix: np.ndarray,
    sample_inc_matrix: np.ndarray,
) -> None:
    np.testing.assert_array_equal(
        convert.adjacency_list_to_adjacency_matrix(sample_adj_list),
        sample_adj_matrix,
    )
    np.testing.assert_array_equal(
        convert.adjacency_list_to_incidence_matrix(sample_adj_list),
        sample_inc_matrix,
    )
    assert (
        convert.adjacency_matrix_to_adjacency_list(sample_adj_matrix) == sample_adj_list
    )
    assert (
        convert.incidence_matrix_to_adjacency_list(sample_inc_matrix) == sample_adj_list
    )
    np.testing.assert_array_equal(
        convert.adjacency_matrix_to_incidence_matrix(sample_adj_matrix),
        sample_inc_matrix,
    )
    np.testing.assert_array_equal(
        convert.incidence_matrix_to_adjacency_matrix(sample_inc_matrix),
        sample_adj_matrix,
    )


def test_empty_and_edgeless_graphs() -> None:
    assert convert.adjacency_list_to_adjacency_matrix([]).shape == (0, 0)
    assert convert.adjacency_list_to_incidence_matrix([]).shape == (0, 0)
    assert (
        convert.adjacency_matrix_to_adjacency_list(np.zeros((0, 0), dtype=np.int64))
        == []
    )
    assert (
        convert.incidence_matrix_to_adjacency_list(np.zeros((0, 0), dtype=np.int64))
        == []
    )

    edgeless_list: AdjacencyList = [[], [], []]
    edgeless_adj = np.zeros((3, 3), dtype=np.int64)
    edgeless_inc = np.zeros((3, 0), dtype=np.int64)

    np.testing.assert_array_equal(
        convert.adjacency_list_to_incidence_matrix(edgeless_list),
        edgeless_inc,
    )
    np.testing.assert_array_equal(
        convert.adjacency_matrix_to_incidence_matrix(edgeless_adj),
        edgeless_inc,
    )
    assert convert.incidence_matrix_to_adjacency_list(edgeless_inc) == edgeless_list
    np.testing.assert_array_equal(
        convert.incidence_matrix_to_adjacency_matrix(edgeless_inc),
        edgeless_adj,
    )


def test_unsorted_adjacency_list_normalized() -> None:
    unsorted = [
        [Neighbor(3, 5), Neighbor(2, 2)],
        [Neighbor(3, 7), Neighbor(1, 2)],
        [Neighbor(2, 7), Neighbor(1, 5)],
    ]
    inc = convert.adjacency_list_to_incidence_matrix(unsorted)
    assert convert.incidence_matrix_to_adjacency_list(inc) == [
        [Neighbor(2, 2), Neighbor(3, 5)],
        [Neighbor(1, 2), Neighbor(3, 7)],
        [Neighbor(1, 5), Neighbor(2, 7)],
    ]


def test_weighted_all_six_conversions() -> None:
    weighted_adj_list: AdjacencyList = [
        [Neighbor(2, 4), Neighbor(3, -2)],
        [Neighbor(1, 4), Neighbor(3, 7)],
        [Neighbor(1, -2), Neighbor(2, 7), Neighbor(4, 3)],
        [Neighbor(3, 3)],
    ]
    weighted_adj_matrix = np.array(
        [
            [0, 4, -2, 0],
            [4, 0, 7, 0],
            [-2, 7, 0, 3],
            [0, 0, 3, 0],
        ],
        dtype=np.int64,
    )
    weighted_inc_matrix = np.array(
        [
            [4, -2, 0, 0],
            [4, 0, 7, 0],
            [0, -2, 7, 3],
            [0, 0, 0, 3],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        convert.adjacency_list_to_adjacency_matrix(weighted_adj_list),
        weighted_adj_matrix,
    )
    np.testing.assert_array_equal(
        convert.adjacency_list_to_incidence_matrix(weighted_adj_list),
        weighted_inc_matrix,
    )
    assert (
        convert.adjacency_matrix_to_adjacency_list(weighted_adj_matrix)
        == weighted_adj_list
    )
    assert (
        convert.incidence_matrix_to_adjacency_list(weighted_inc_matrix)
        == weighted_adj_list
    )
    np.testing.assert_array_equal(
        convert.adjacency_matrix_to_incidence_matrix(weighted_adj_matrix),
        weighted_inc_matrix,
    )
    np.testing.assert_array_equal(
        convert.incidence_matrix_to_adjacency_matrix(weighted_inc_matrix),
        weighted_adj_matrix,
    )


@pytest.mark.parametrize(
    "bad_list",
    [
        [[Neighbor(0, 1)]],
        [[Neighbor(1, 1)]],
        [[Neighbor(3, 1)], []],
        [[Neighbor(2, 1), Neighbor(2, 1)], [Neighbor(1, 1)]],
        [[(True, 1)], []],
        [[(2, True)], [(1, 1)]],
        [[Neighbor(2, 0)], [Neighbor(1, 0)]],
        [[2], [1]],
    ],
)
def test_invalid_adjacency_list(bad_list: Any) -> None:
    with pytest.raises(ValueError):
        convert.adjacency_list_to_adjacency_matrix(bad_list)


@pytest.mark.parametrize(
    "bad_matrix",
    [
        [[0, 1]],
        [[1, 0], [0, 0]],
        [0, 1],
    ],
)
def test_invalid_adjacency_matrix(bad_matrix: Any) -> None:
    with pytest.raises(ValueError):
        convert.adjacency_matrix_to_adjacency_list(bad_matrix)


def test_directed_weighted_conversions() -> None:
    digraph_list: AdjacencyList = [
        [Neighbor(2, 4), Neighbor(3, -2)],
        [Neighbor(1, -1)],
        [Neighbor(2, 5)],
    ]
    digraph_matrix = np.array(
        [
            [0, 4, -2],
            [-1, 0, 0],
            [0, 5, 0],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        convert.adjacency_list_to_adjacency_matrix(digraph_list),
        digraph_matrix,
    )
    assert convert.adjacency_matrix_to_adjacency_list(digraph_matrix) == digraph_list

    with pytest.raises(ValueError):
        convert.adjacency_list_to_incidence_matrix(digraph_list)
    with pytest.raises(ValueError):
        convert.adjacency_matrix_to_incidence_matrix(digraph_matrix)


@pytest.mark.parametrize(
    "bad_inc",
    [
        [[1], [0]],
        [[1, 1], [1, 1]],
        [[2], [0]],
        [[2], [3]],
        [1, 1],
    ],
)
def test_invalid_incidence_matrix(bad_inc: Any) -> None:
    with pytest.raises(ValueError):
        convert.incidence_matrix_to_adjacency_list(bad_inc)
