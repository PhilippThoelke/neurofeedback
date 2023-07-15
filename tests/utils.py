import time
from typing import Any, Dict, List, Optional

import numpy as np

from neurofeedback.utils import Data, DataIn, DataType, Processor


class DummyProcessor(Processor):
    NAME = "dummy"
    SUPPORTED_DTYPES = DataType.ALL

    def __init__(
        self, *input_addresses: str, illegal_modify_data: bool = False, **kwargs
    ):
        super().__init__(*input_addresses, **kwargs)
        self.illegal_modify_data = illegal_modify_data

    def process(self, data: Dict[str, Data]) -> Data:
        if self.illegal_modify_data:
            # this should raise an error in the superclass because we are modifying the data
            d = Data("/illegal-feature", "dummy-value", DataType.STRING)
            data[d.address] = d
        return Data("feature", "dummy-value", DataType.STRING)


class DummyStream(DataIn):
    """
    Dummy stream for testing purposes.

    Parameters:
        address (str): the address of the stream
        data (str): the type of data to generate, one of "normal", "zeros", "arange", "exp"
        n_channels (int): number of channels to generate
        sfreq (float): sampling frequency
        **kwargs: additional keyword arguments to pass to the DataIn constructor
    """

    def __init__(
        self,
        address: str = "dummy",
        data: str = "normal",
        n_channels: int = 5,
        sfreq: float = 100,
        **kwargs,
    ):
        super().__init__(address=address, **kwargs)
        assert data in ["normal", "zeros", "arange", "exp"]

        self.data = data
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.last_receive = None

        # fill the buffer so we don't need to wait for the first receive
        self.update({})

    @property
    def info(self) -> Dict[str, Any]:
        return dict(sfreq=self.sfreq, ch_type=["misc"] * self.n_channels)

    def receive(self) -> np.ndarray:
        if self.last_receive is None:
            n_samples = int(self.buffer_seconds * self.sfreq)
        else:
            n_samples = int((time.time() - self.last_receive) * self.sfreq)
        self.last_receive = time.time()

        if self.data == "normal":
            dat = np.random.normal(size=(self.n_channels, n_samples))
        elif self.data == "zeros":
            dat = np.zeros((self.n_channels, n_samples))
        elif self.data == "arange":
            dat = (
                np.arange(n_samples)
                .repeat(self.n_channels)
                .reshape(n_samples, self.n_channels)
                .T
            )
        elif self.data == "exp":
            dat = (
                np.exp(np.arange(n_samples) / self.sfreq)
                .repeat(self.n_channels)
                .reshape(n_samples, self.n_channels)
                .T
            )
        return dat.astype(np.float32)
