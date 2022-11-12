import pytest
from tests import TEST_FOLDER
from tensor_check import checker


def test_ones():
    with pytest.raises(AssertionError):
        checker.check_file(TEST_FOLDER / "construct.py")
