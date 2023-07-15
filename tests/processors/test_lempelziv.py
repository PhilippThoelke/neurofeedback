import numpy as np
import pytest
from antropy import lziv_complexity

from neurofeedback.processors import LempelZiv
from tests.utils import DummyStream


@pytest.mark.parametrize("binarize", ["mean", "median"])
def test_process(binarize):
    dat = DummyStream()
    proc = LempelZiv(label="lziv", binarize_mode=binarize)

    # make sure the processor returns a float with the correct label
    result = {}
    proc({"dummy": dat}, result, {})

    assert "/dummy/lziv" in result
    assert isinstance(result["/dummy/lziv"], float)


@pytest.mark.parametrize("binarize", ["mean", "median"])
@pytest.mark.parametrize("data", ["arange", "zeros", "exp"])
def test_output(binarize, data):
    dat = DummyStream(data=data, n_channels=1)
    proc = LempelZiv(label="lziv", binarize_mode=binarize)

    # compute expected result
    raw = np.array(dat.buffer)[:, 0]
    if binarize == "mean":
        binarized = raw >= np.mean(raw)
    elif binarize == "median":
        binarized = raw >= np.median(raw)

    # compare processor output with antropy
    result = {}
    proc({"dummy": dat}, result, {})
    assert result["/dummy/lziv"] == lziv_complexity(binarized, normalize=True)
