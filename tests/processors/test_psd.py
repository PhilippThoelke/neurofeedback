import pytest

from neurofeedback.processors import PSD
from tests.utils import DummyStream


@pytest.mark.parametrize("band", ["delta", "theta", "alpha", "beta", "gamma"])
def test_process(band):
    dat = DummyStream()
    proc = PSD(label=band)

    # make sure the processor returns a float with the correct label
    result = {}
    proc({"dummy": dat}, result, {})

    assert f"/dummy/{band}" in result
    assert isinstance(result[f"/dummy/{band}"], float)
