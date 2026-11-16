from typing import Any, cast

import numpy as np
import pytest

from graphs import utils


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
