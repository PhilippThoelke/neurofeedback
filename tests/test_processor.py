import pytest

from tests.utils import DummyProcessor


def test_suffix():
    p = DummyProcessor(".*", output_suffix="suffix")
    data = {}
    p.update(data)
    assert "/dummy-suffix/feature" in data, "Suffix not added to output address"


def test_duplicate_output_address():
    p1 = DummyProcessor(".*")
    p2 = DummyProcessor(".*")

    data = dict()
    p1.update(data)
    with pytest.raises(RuntimeError):
        # this should raise because p1's and p2's features have the same address
        p2.update(data)


def test_dirty_duplicate_output_address():
    p1 = DummyProcessor(".*")
    p2 = DummyProcessor(".*")

    data = dict()
    p1.update(data)
    assert "/dummy/feature" in data, "Missing feature from p1"
    data["/dummy/feature"].dirty = True

    # this should NOT raise an error because we set p1's output to dirty
    p2.update(data)


def test_update_modify():
    p = DummyProcessor(".*", illegal_modify_data=True)
    with pytest.raises(RuntimeError):
        # this should raise because we are modifying the data dictionary directly
        p.update({})
