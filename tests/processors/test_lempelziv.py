import numpy as np
import pytest
from antropy import lziv_complexity

from neurofeedback.processors import LempelZiv
from neurofeedback.utils import DataType
from tests.utils import DummyStream


@pytest.mark.parametrize("binarize", ["mean", "median"])
@pytest.mark.parametrize("lziv_name", ["lziv", "lempel-ziv"])
@pytest.mark.parametrize("dummy_name", ["dummy", "dumdum"])
def test_process(binarize, lziv_name, dummy_name):
    dat = DummyStream(address=dummy_name)
    proc = LempelZiv(output_address=lziv_name, binarize_mode=binarize, reduce=None)

    # populate the data dict
    data = {}
    dat.update(data)
    proc.update(data)

    # make sure we got the correct output addresses
    for i in range(dat.n_channels):
        addr = f"/{lziv_name}/{dummy_name}/ch{i}"
        assert addr in data, f"missing channel {i}"
        assert data[addr].dtype == DataType.FLOAT, f"wrong dtype for channel {i}"


@pytest.mark.parametrize("binarize", ["mean", "median"])
@pytest.mark.parametrize("data", ["arange", "zeros", "exp"])
def test_output(binarize, data):
    dat = DummyStream(address="dummy", data=data, n_channels=1)
    proc = LempelZiv(output_address="lziv", binarize_mode=binarize)

    # compute expected result
    raw = np.array(dat.buffer)[:, 0]
    if binarize == "mean":
        binarized = raw >= np.mean(raw)
    elif binarize == "median":
        binarized = raw >= np.median(raw)

    # compare processor output with antropy
    result = {}
    dat.update(result)
    proc.update(result)
    assert result["/lziv/dummy/ch0"].value == lziv_complexity(binarized, normalize=True)
