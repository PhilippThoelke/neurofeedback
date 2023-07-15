import time
import warnings
from os.path import exists
from typing import Dict, List

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyedflib import EdfWriter, highlevel
from pythonosc import osc_bundle_builder
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.udp_client import UDPClient

from neurofeedback.utils import Data, DataIn, DataOut, DataType, fmt_address


class PlotRaw(DataOut):
    """
    Real-time visualization of the raw EEG buffer.

    ONLY USE VISUALIZATION FOR DEBUGGING AS IT CAN SIGNIFICANTLY SLOW DOWN PROCESSING

    Parameters:
        data_in_name (str): name of the input stream to visualize
        scaling (float): scaling factor for the visualization of raw EEG
    """

    def __init__(self, data_in_name=None, scaling: float = 5e-3):
        self.data_in_name = data_in_name
        self.scaling = scaling

        # initialize figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Raw EEG buffer")
        self.ax.set_xlabel("time (s)")
        self.fig.show()
        self.line_plots = None

        self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.fig_size = self.fig.get_size_inches()

    def update(
        self,
        data_in: Dict[str, DataIn],
        processed: Dict[str, float],
    ):
        """
        Update the plot of the raw EEG signal.

        Parameters:
            data_in (Dict[str, DataIn]): list of input streams
            processed (Dict[str, float]): dictionary of extracted, normalized features
        """
        assert (
            self.data_in_name is not None or len(data_in) == 1
        ), "data_in_name must be specified if multiple input streams are used"
        if self.data_in_name is None:
            self.data_in_name = list(data_in.keys())[0]

        raw = np.array(data_in[self.data_in_name].buffer).T
        xs = np.arange(-raw.shape[1], 0) / data_in[self.data_in_name].info["sfreq"]
        raw = raw * self.scaling + np.arange(raw.shape[0])[:, None]

        if (self.fig_size != self.fig.get_size_inches()).any():
            # hide lines
            for line in self.line_plots:
                line.set_visible(False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # recapture background
            self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig_size = self.fig.get_size_inches()

            # unhide lines
            for line in self.line_plots:
                line.set_visible(True)

        # restore background
        self.fig.canvas.restore_region(self.background_buffer)

        # update line plots
        if self.line_plots is None:
            xs = xs[None].repeat(raw.shape[0], axis=0).reshape(raw.shape)
            self.line_plots = self.ax.plot(xs.T, raw.T, c="0", linewidth=0.7)
            self.ax.set_yticks(
                np.arange(raw.shape[0]), data_in[self.data_in_name].info["ch_names"]
            )
            self.fig_size = None
        else:
            for i, line in enumerate(self.line_plots):
                line.set_data(xs, raw[i])
                self.ax.draw_artist(line)

        # rescale axes
        self.ax.relim()
        self.ax.autoscale_view()

        # redraw the ax
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()


class PlotProcessed(DataOut):
    """
    Real-time visualization of extracted features in a bar plot.

    ONLY USE VISUALIZATION FOR DEBUGGING AS IT CAN SIGNIFICANTLY SLOW DOWN PROCESSING
    """

    def __init__(self):
        # initialize figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Processed EEG features")
        self.ax.set_xlabel("features")
        self.ax.set_ylim(-3, 3)
        self.fig.show()
        self.bar_plots = None

        self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.fig_size = self.fig.get_size_inches()

    def update(
        self,
        data_in: Dict[str, DataIn],
        processed: Dict[str, float],
    ):
        """
        Update the plot of extracted features.

        Parameters:
            data_in (Dict[str, DataIn]): list of input streams
            processed (Dict[str, float]): dictionary of extracted, normalized features
        """
        if (self.fig_size != self.fig.get_size_inches()).any():
            # hide bars
            for bar in self.bar_plots:
                bar.set_visible(False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # recapture background
            self.background_buffer = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig_size = self.fig.get_size_inches()

            # unhide bars
            for bar in self.bar_plots:
                bar.set_visible(True)

        # restore background
        self.fig.canvas.restore_region(self.background_buffer)

        # update line plots
        values = [p if isinstance(p, float) else 0.0 for p in processed.values()]
        if self.bar_plots is None:
            xs = range(len(processed))
            self.bar_plots = self.ax.bar(
                xs, values, color=[f"C{i}" for i in range(len(processed))]
            )
            self.ax.set_xticks(xs, processed.keys(), rotation=70, ha="right", x=0.9)
            self.ax.grid()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig_size = None
        else:
            for bar, val in zip(self.bar_plots, values):
                bar.set_height(val)
                self.ax.draw_artist(bar)

            if self.ax.get_xlim()[0] == 0:
                self.ax.relim()
                self.ax.autoscale(axis="x")
                self.fig.canvas.draw_idle()

        # redraw the ax
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()


class OSCStream(DataOut):
    SUPPORTED_DTYPES = (
        DataType.FLOAT | DataType.STRING | DataType.ARRAY_1D | DataType.IMAGE
    )
    """
    Stream extracted features via OSC. The OSC addresses are the names of the extracted
    features (i.e. the keys in the processed dictionary), prefixed by address_prefix.

    Parameters:
        ip (str): target IP address
        port (int): target port
        address_prefix (str): prefix for the OSC address
    """

    def __init__(self, ip: str, port: int, address_prefix: str = ""):
        self.address_prefix = address_prefix
        self.client = UDPClient(ip, port)

    def output(self, data: List[Data]):
        """
        Send the extracted features to the target OSC server. The processed dictionary
        gets combined into a single OSC Bundle with a different OSC address per feature.

        Parameters:
            data (List[Data]): list of data streams
        """
        # Initialize an empty bundle
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

        for d in data:
            # Create a new message
            msg = OscMessageBuilder(fmt_address(f"{self.address_prefix}/{d.address}"))
            if d.dtype in (DataType.ARRAY_1D | DataType.IMAGE):
                # TODO: make sure larger messages are handled correctly
                warnings.warn("Large messages are currently broken.")

                # python-osc automatically handles lists
                msg.add_arg(d.value.tolist())
            else:
                # let OSC guess the type
                msg.add_arg(d.value)

            # Add the message to the bundle
            bundle.add_content(msg.build())

            # If the bundle size is too large, send it and start a new one
            # TODO: make sure 65507 is the correct limit
            if sum(len(msg.dgram) for msg in bundle._contents) > 1472:
                self.client.send(bundle.build())
                bundle = osc_bundle_builder.OscBundleBuilder(
                    osc_bundle_builder.IMMEDIATELY
                )

        # Send the remaining bundle
        self.client.send(bundle.build())


class RawToFile(DataOut):
    """
    Stream raw EEG data to a CSV or EDF file on disk.

    Note: When saving in EDF format, data is saved in chunks of one second, meaning that
    the last second of data after interrupting the processing loop might not appear in
    the file on disk.

    Parameters:
        fname (str): the file name (should end in .csv or .edf)
        data_in_name (str): name of the input stream to visualize
        overwrite (bool): if False, raise an error if the specified file already exists
    """

    def __init__(self, fname: str, data_in_name=None, overwrite: bool = False):
        if not overwrite and exists(fname):
            raise FileExistsError(
                f'The file "{fname}" already exists. You can set '
                "overwrite=True if you want to allow overwriting."
            )

        self.fname = fname
        self.file_type = fname.split(".")[-1].lower()

        # initialize file type specific attributes
        if self.file_type == "csv":
            self.header_done = False
            self.start_time = None
        elif self.file_type == "edf":
            self.writer = None
            self.chunk_buffer = None
            self.chunk_idx = 0
        else:
            raise ValueError(
                f'Unsupported file type "{self.file_type}". '
                "The file name should end in .csv or .edf."
            )

    def update(
        self,
        data_in: Dict[str, DataIn],
        processed: Dict[str, float],
    ):
        """
        Receives new raw data points and stores them in an EDF file in 1 second chunks.

        Parameters:
            data_in (Dict[str, DataIn]): list of input streams
            processed (Dict[str, float]): dictionary of extracted, normalized features
            n_samples_received (int): number of new samples in the raw buffer
        """
        assert (
            self.data_in_name is not None or len(data_in) == 1
        ), "data_in_name must be specified if multiple input streams are used"

        raw = np.array(data_in[self.data_in_name].buffer).T

        if self.file_type == "csv":
            self._update_csv(
                raw,
                data_in[self.data_in_name].info,
                processed,
                data_in[self.data_in_name].n_samples_received,
            )
        elif self.file_type == "edf":
            self._update_edf(
                raw,
                data_in[self.data_in_name].info,
                processed,
                data_in[self.data_in_name].n_samples_received,
            )

    def _update_csv(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        n_samples_received: int,
    ):
        file_mode = "a"
        if not self.header_done:
            self.start_time = time.time()
            file_mode = "w"

        # create a time-based index for the newly acquired samples
        index = (
            time.time()
            - self.start_time
            - np.arange(n_samples_received)[::-1] / info["sfreq"]
        )
        # create a DataFrame with newly acquired raw EEG samples
        df = pd.DataFrame(
            raw[:, -n_samples_received:].T, columns=info["ch_names"], index=index
        )
        df.to_csv(self.fname, mode=file_mode, header=not self.header_done)

        self.header_done = True

    def _update_edf(
        self,
        raw: np.ndarray,
        info: mne.Info,
        processed: Dict[str, float],
        n_samples_received: int,
    ):
        if self.writer is None:
            # initialize the chunk buffer to hold 1 second of data
            self.chunk_buffer = np.zeros((info["nchan"], int(info["sfreq"])))

            # initialize the EDF writer
            self.writer = EdfWriter(self.fname, info["nchan"])
            # write headers
            self.writer.setHeader(highlevel.make_header())
            self.writer.setSignalHeaders(
                highlevel.make_signal_headers(
                    info["ch_names"], sample_frequency=info["sfreq"]
                )
            )

        n_to_process = n_samples_received
        while n_to_process > 0:
            # find how many samples we can write into the chunk buffer
            n_samples = min(self.chunk_buffer.shape[1] - self.chunk_idx, n_to_process)
            segment = self.chunk_buffer[:, self.chunk_idx : self.chunk_idx + n_samples]

            # write newly received samples into the buffer
            if n_samples == n_to_process:
                segment[:] = raw[:, -n_to_process:]
            else:
                segment[:] = raw[:, -n_to_process : -n_to_process + n_samples]
            self.chunk_idx += n_samples

            # make sure the chunk index never exceeds the buffer length
            assert (
                self.chunk_idx <= self.chunk_buffer.shape[1]
            ), "Internal Error: Chunk index exceeded buffer size"

            if self.chunk_idx == self.chunk_buffer.shape[1]:
                # chunk buffer is full, write samples to disk
                self.writer.writeSamples(self.chunk_buffer)
                self.chunk_buffer[:] = 0
                self.chunk_idx = 0

            n_to_process -= n_samples


class ProcessedToFile(DataOut):
    """
    Stream extracted features to a CSV file on disk. The first (index-)column contains the sampling
    time in seconds after data acquisition was started. Columns are named according to the feature
    names.

    Parameters:
        fname (str): the file name of the resulting EDF file
        overwrite (bool): if False, raise an error if the specified file already exists
    """

    def __init__(self, fname: str, overwrite: bool = False):
        if not overwrite and exists(fname):
            raise FileExistsError(
                f'The file "{fname}" already exists. You can set '
                "overwrite=True if you want to allow overwriting."
            )
        assert (
            fname.split(".")[-1].lower() == "csv"
        ), "Expected the file ending to be .csv"

        self.fname = fname
        self.header_done = False
        self.start_time = None

    def update(
        self,
        data_in: Dict[str, DataIn],
        processed: Dict[str, float],
    ):
        """
        Appends newly extracted features as a new row to a CSV file. If the file doesn't exist yet
        also creates the file and writes the header.

        Parameters:
            data_in (Dict[str, DataIn]): list of input streams
            processed (Dict[str, float]): dictionary of extracted, normalized features
        """
        file_mode = "a"
        if not self.header_done:
            self.start_time = time.time()
            file_mode = "w"

        # create a DataFrame out of the features and append it to the CSV file
        df = pd.DataFrame(processed, index=[time.time() - self.start_time])
        df.to_csv(self.fname, mode=file_mode, header=not self.header_done)

        self.header_done = True
