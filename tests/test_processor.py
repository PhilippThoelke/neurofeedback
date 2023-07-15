import pytest

from tests.utils import DummyProcessor


def test_duplicate_output_address():
    p1 = DummyProcessor(".*", output_address="dummy")
    p2 = DummyProcessor(".*", output_address="dummy")
    data = dict()

    p1.update(data)
    with pytest.raises(RuntimeError):
        # this should raise an error because p1 and p2 have the same output address
        p2.update(data)
