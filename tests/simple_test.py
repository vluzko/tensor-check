"""Basic tests that the checker runs correctly."""
from pathlib import Path
from tensor_check import checker, tc_types

TEST_FOLDER = Path(__file__).parent / "test_files"


def test_integers():
    f = TEST_FOLDER / "arithmetic.py"
    context = checker.check_file(f)
    x_type = context.lookup_name("x")
    assert isinstance(x_type, tc_types.InternalInt)
    assert x_type.constraints == [tc_types.Equal(tc_types.Self(), 1)]
    y_type = context.lookup_name("y")
    assert isinstance(y_type, tc_types.InternalInt)
    assert y_type.constraints == [tc_types.Equal(tc_types.Self(), 2)]
    z_type = context.lookup_name("z")
    assert isinstance(z_type, tc_types.InternalInt)
    assert z_type.constraints == [tc_types.Equal(tc_types.Self(), "x + y")]
