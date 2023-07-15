import pytest

from neurofeedback.processors import PSD
from neurofeedback.utils import DataType
from tests.utils import DummyStream


@pytest.mark.parametrize("band", ["delta", "theta", "alpha", "beta", "gamma"])
@pytest.mark.parametrize("psd_name", ["psd", "power"])
@pytest.mark.parametrize("dummy_name", ["dummy", "dumdum"])
def test_process(band, psd_name, dummy_name):
    dat = DummyStream(address=dummy_name)
    proc = PSD(output_suffix=psd_name, band=band, reduce=None)

    # make sure the processor returns a float with the correct label
    data = {}
    dat.update(data)
    proc.update(data)

    # make sure we got the correct output addresses
    for i in range(dat.n_channels):
        addr = f"/{psd_name}/{dummy_name}/ch{i}"
        assert addr in data, f"missing channel {i}"
        assert data[addr].dtype == DataType.FLOAT, f"wrong dtype for channel {i}"
