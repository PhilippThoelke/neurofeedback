import pytest

from tests.utils import DummyProcessor


def test_duplicate_output_address():
    p1 = DummyProcessor(".*", output_address="dummy")
    p2 = DummyProcessor(".*", output_address="dummy")

    data = dict()
    p1.update(data)
    with pytest.raises(RuntimeError):
        # this should raise because p1's and p2's features have the same address
        p2.update(data)


def test_dirty_duplicate_output_address():
    p1 = DummyProcessor(".*", output_address="dummy")
    p2 = DummyProcessor(".*", output_address="dummy")

    data = dict()
    p1.update(data)
    data["/dummy/feature"].dirty = True

    # this should NOT raise an error because we set p1's output to dirty
    p2.update(data)
