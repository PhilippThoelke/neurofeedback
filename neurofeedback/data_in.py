import threading
import time
from typing import Any, Dict, List, Optional, Union

import mne
import numpy as np
import serial
import serial.tools.list_ports
from mne.datasets import eegbci
from mne.io import BaseRaw, concatenate_raws, read_raw
from mne_realtime import LSLClient, MockLSLStream

from neurofeedback.utils import MNE_CHANNEL_TYPE_MAPPING, DataIn

mne.set_log_level(False)


class SerialStream(DataIn):
    def __init__(self, sfreq: int, address: str = "serial", **kwargs):
        super().__init__(address=address, **kwargs)
        self.sfreq = sfreq
        self.serial_buffer = []
        self.buffer_times = []
        self.last_processed_time = None
        self.lock = threading.Lock()
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.serial_thread.start()

    def read_serial(self):
        ser = serial.Serial(SerialStream.detect_serial_port(), 115200, timeout=1)
        if not ser.isOpen():
            ser.open()

        ESC = b"\xDB"
        END = b"\xC0"
        ESC_END = b"\xDC"
        ESC_ESC = b"\xDD"

        packet = bytearray()
        while True:
            c = ser.read()
            if c == END:
                if packet:  # ignore empty packets
                    if len(packet) == 2:
                        value = packet[0] << 8 | packet[1]
                        current_time = time.time()
                        with self.lock:
                            self.serial_buffer.append(value)
                            self.buffer_times.append(current_time)
                    else:
                        print("Invalid packet received")
                    packet = bytearray()
            elif c == ESC:
                c = ser.read()
                if c == ESC_END:
                    packet.append(0xC0)
                elif c == ESC_ESC:
                    packet.append(0xDB)
                else:
                    print("Invalid escape sequence")
            elif len(c) > 0:
                packet.append(c[0])

    @property
    def info(self) -> Dict[str, Any]:
        return dict(ch_names=["serial"], ch_type=["bio"], sfreq=self.sfreq)

    def receive(self) -> np.ndarray:
        with self.lock:
            # Copy the buffer and times list
            data = np.array(self.serial_buffer)
            times = np.array(self.buffer_times)

            if len(data) < 2:
                return None

            # Calculate new times for interpolation
            if self.last_processed_time is None:
                new_times_start = times[0]
            else:
                new_times_start = self.last_processed_time

            # Calculate the number of samples based on the sampling frequency and the time span
            num_samples = int((times[-1] - new_times_start) * self.sfreq)
            if num_samples == 0:
                return None

            # Create the new time array
            new_times = np.linspace(new_times_start, times[-1], num_samples)

            # Clear the buffer and times list
            self.serial_buffer.clear()
            self.buffer_times.clear()

        # Resample the data to a common sampling frequency
        new_data = np.interp(new_times, times, data)

        # Update the last processed time
        self.last_processed_time = new_times[-1]
        return new_data[None]

    def detect_serial_port():
        ports = serial.tools.list_ports.comports()
        print(ports)
        for port in ports:
            if "Arduino" in port.description or "Serial" in port.description:
                return port.device
        return None


class EEGStream(DataIn):
    """
    Incoming LSL stream with raw EEG data.

    Parameters:
        host (str): the LSL stream's hostname
        port (int): the LSL stream's port (if None, use the LSLClient's default port)
        address (str): the address of the resulting data stream
        **kwargs: additional arguments to pass to the DataIn constructor
    """

    def __init__(
        self, host: str, port: Optional[int] = None, address: str = "eeg", **kwargs
    ):
        super().__init__(address=address, **kwargs)

        # start LSL client
        self.client = LSLClient(host=host, port=port)
        self.client.start()
        self.data_iterator = self.client.iter_raw_buffers()

    @property
    def info(self) -> Dict[str, Any]:
        info = dict(self.client.get_measurement_info())
        info["ch_type"] = [MNE_CHANNEL_TYPE_MAPPING[ch["kind"]] for ch in info["chs"]]
        return info

    def receive(self) -> np.ndarray:
        data = next(self.data_iterator)
        if data.size == 0:
            return None
        return data


class EEGRecording(EEGStream):
    """
    Stream previously recorded EEG from a file. The data is loaded from the file and streamed
    to a mock LSL stream, which is then accessed via the EEGStream parent-class.

    Parameters:
        raw (str, BaseRaw): file-name of a raw EEG file or an instance of mne.io.BaseRaw
        address (str): the address of the resulting data stream
        **kwargs: additional arguments to pass to the EEGStream constructor
    """

    def __init__(self, raw: Union[str, BaseRaw], address: str = "eeg-file", **kwargs):
        # load raw EEG data
        if not isinstance(raw, BaseRaw):
            raw = read_raw(raw)
        raw.load_data().pick("eeg")

        # start the mock LSL stream to serve the EEG recording
        host = "mock-eeg-stream"
        self.mock_stream = MockLSLStream(host, raw, "eeg")
        self.mock_stream.start()

        # start the LSL client
        super().__init__(host=host, address=address, **kwargs)

    @staticmethod
    def make_eegbci(
        address: str = "eegbci",
        subjects: Union[int, List[int]] = 1,
        runs: Union[int, List[int]] = [1, 2],
        **kwargs,
    ):
        """
        Static utility function to instantiate an EEGRecording instance using
        the PhysioNet EEG BCI dataset. This function automatically downloads the
        dataset if it is not present.
        See https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#mne-datasets-eegbci-load-data
        for information about the dataset and a description of different runs.

        Parameters:
            address (str): the address of the resulting data stream
            subjects (int, List[int]): which subject(s) to load data from
            runs (int, List[int]): which run(s) to load from the corresponding subject
            **kwargs: additional arguments to pass to the EEGRecording constructor
        """
        raw = concatenate_raws([read_raw(p) for p in eegbci.load_data(subjects, runs)])
        eegbci.standardize(raw)
        return EEGRecording(raw, address=address, **kwargs)
