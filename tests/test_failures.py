from dim_checker import checker
from pytest import raises


def test_fail_add():
    contents = """
x = torch.ones((4, 4))
y = torch.ones((5, 5))
x + y"""
    with raises(TypeError):
        checker.check(contents)