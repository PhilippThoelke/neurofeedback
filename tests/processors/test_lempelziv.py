import numpy as np
import pytest
from antropy import lziv_complexity

from neurofeedback.processors import LempelZiv
from neurofeedback.utils import DataType
from tests.utils import DummyStream


@pytest.mark.parametrize("binarize", ["mean", "median"])
@pytest.mark.parametrize("dummy_name", ["dummy", "dumdum"])
def test_process(binarize, dummy_name):
    dat = DummyStream(address=dummy_name)
    p = LempelZiv(binarize_mode=binarize, reduce=None)

    # populate the data dict
    data = {}
    dat.update(data)
    p.update(data)

    # make sure we got the correct output addresses
    for i in range(dat.n_channels):
        addr = f"/{p.NAME}/{dummy_name}/ch{i}"
        assert addr in data, f"missing channel {i}"
        assert data[addr].dtype == DataType.FLOAT, f"wrong dtype for channel {i}"


@pytest.mark.parametrize("binarize", ["mean", "median"])
@pytest.mark.parametrize("data", ["arange", "zeros", "exp"])
def test_output(binarize, data):
    dat = DummyStream(address="dummy", data=data, n_channels=1)
    p = LempelZiv(binarize_mode=binarize)

    # compute expected result
    raw = np.array(dat.buffer)[:, 0]
    if binarize == "mean":
        binarized = raw >= np.mean(raw)
    elif binarize == "median":
        binarized = raw >= np.median(raw)

    # compare processor output with antropy
    result = {}
    dat.update(result)
    p.update(result)
    assert result[f"/{p.NAME}/dummy/ch0"].value == lziv_complexity(
        binarized, normalize=True
    ), "Value did not match the reference implementation"
