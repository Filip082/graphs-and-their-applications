from typing import Any

import numpy as np
import pytest

from graphs import convert


@pytest.fixture
def sample_adj_list() -> list[list[int]]:
    return [
        [1, 2],
        [0, 2],
        [0, 1, 3],
        [2],
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
    sample_adj_list: list[list[int]],
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

    edgeless_list: list[list[int]] = [[], [], []]
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
    unsorted = [[2, 1], [2, 0], [1, 0]]
    inc = convert.adjacency_list_to_incidence_matrix(unsorted)
    assert convert.incidence_matrix_to_adjacency_list(inc) == [
        [1, 2],
        [0, 2],
        [0, 1],
    ]


@pytest.mark.parametrize(
    "bad_list",
    [
        [[0]],
        [[1], []],
        [[2], []],
        [[1, 1], [0]],
        [[True], []],
    ],
)
def test_invalid_adjacency_list(bad_list: list[list[int]]) -> None:
    with pytest.raises(ValueError):
        convert.adjacency_list_to_adjacency_matrix(bad_list)


@pytest.mark.parametrize(
    "bad_matrix",
    [
        [[0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 2], [2, 0]],
        [0, 1],
    ],
)
def test_invalid_adjacency_matrix(bad_matrix: Any) -> None:
    with pytest.raises(ValueError):
        convert.adjacency_matrix_to_adjacency_list(bad_matrix)


@pytest.mark.parametrize(
    "bad_inc",
    [
        [[1], [0]],
        [[1, 1], [1, 1]],
        [[2], [0]],
        [1, 1],
    ],
)
def test_invalid_incidence_matrix(bad_inc: Any) -> None:
    with pytest.raises(ValueError):
        convert.incidence_matrix_to_adjacency_list(bad_inc)
