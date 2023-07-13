import colorsys
import re
import threading
from abc import ABC, abstractmethod, abstractproperty
from collections import deque
from dataclasses import dataclass
from enum import Flag
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import webcolors
from biotuner.biocolors import audible2visible, scale2freqs, wavelength_to_rgb
from biotuner.bioelements import (
    ALL_ELEMENTS,
    convert_to_nm,
    find_matching_spectral_lines,
    hertz_to_nm,
)
from biotuner.biotuner_object import compute_biotuner, dyad_similarity, harmonic_tuning
from biotuner.harmonic_connectivity import harmonic_connectivity
from biotuner.metrics import tuning_cons_matrix
from gtts import gTTS
from mne.io.constants import FIFF
from playsound import playsound

MNE_CHANNEL_TYPE_MAPPING = {
    FIFF.FIFFV_BIO_CH: "bio",
    FIFF.FIFFV_MEG_CH: "meg",
    FIFF.FIFFV_REF_MEG_CH: "ref_meg",
    FIFF.FIFFV_EEG_CH: "eeg",
    FIFF.FIFFV_MCG_CH: "mcg",
    FIFF.FIFFV_STIM_CH: "stim",
    FIFF.FIFFV_EOG_CH: "eog",
    FIFF.FIFFV_EMG_CH: "emg",
    FIFF.FIFFV_ECG_CH: "ecg",
    FIFF.FIFFV_MISC_CH: "misc",
    FIFF.FIFFV_RESP_CH: "resp",
    FIFF.FIFFV_SEEG_CH: "seeg",
    FIFF.FIFFV_DBS_CH: "dbs",
    FIFF.FIFFV_SYST_CH: "syst",
    FIFF.FIFFV_ECOG_CH: "ecog",
    FIFF.FIFFV_IAS_CH: "ias",
    FIFF.FIFFV_EXCI_CH: "exci",
    FIFF.FIFFV_DIPOLE_WAVE: "dipole_wave",
    FIFF.FIFFV_GOODNESS_FIT: "goodness_fit",
    FIFF.FIFFV_FNIRS_CH: "fnirs",
    FIFF.FIFFV_GALVANIC_CH: "galvanic",
    FIFF.FIFFV_TEMPERATURE_CH: "temperature",
    FIFF.FIFFV_EYETRACK_CH: "eyetrack",
}


class DataType(Flag):
    FLOAT = 1
    STRING = 2
    ARRAY_1D = 4
    IMAGE = 8
    RAW_CHANNEL = 16


@dataclass
class Data:
    address: str
    value: Any
    dtype: DataType

    def __post_init__(self):
        self.address = fmt_address(self.address)


@dataclass
class RawData:
    """
    Container for a single raw data channel with info dictionary.
    The info must at minimum contain the sampling frequency ('sfreq') and
    channel types ('ch_types'). If no channel name is provided, it will
    automatically be set to 'ch_name': 'ch{ch_idx}'.

    If the data is multi-channel, only the channel with index 'ch_idx' will be
    selected.

    Parameters:
        data (np.ndarray): single channel with raw time series data as a 1D array
        info (dict): the info dictionary for this channel
        ch_idx (int): the index of this channel in the raw data stream
    """

    data: np.ndarray
    info: dict
    ch_idx: int

    def __post_init__(self):
        if self.data.ndim > 1:
            # select channel from info
            updated_info = {}
            for key, value in self.info.items():
                if isinstance(value, list) and len(value) == self.data.shape[1]:
                    updated_info[key] = value[self.ch_idx]
                else:
                    updated_info[key] = value
            self.info = updated_info

            # select channel from data
            self.data = self.data[:, self.ch_idx]

        # check data and info
        assert isinstance(self.data, np.ndarray), "RawData.data must be a numpy array"
        assert self.data.ndim == 1, "RawData.data must be a 1D array"
        assert "sfreq" in self.info, "RawData.info must contain 'sfreq' key"
        assert "ch_types" in self.info, "RawData.info must contain 'ch_types' key"

        # set channel name if not provided
        if "ch_name" not in self.info:
            if "ch_names" in self.info:
                self.info["ch_name"] = self.info["ch_names"]
            else:
                self.info["ch_name"] = f"ch{self.ch_idx}"


class DataIn(ABC):
    """
    Abstract data input stream. Derive from this class to implement new input streams.

    Parameters:
        address (str): the address of this input stream
        buffer_seconds (float): the number of seconds to buffer incoming data
        rescale (float): the factor to rescale the incoming data with
    """

    def __init__(self, address: str, buffer_seconds: float = 5, rescale: float = 1):
        self.address = address
        self.buffer_seconds = buffer_seconds
        self.rescale = rescale

        self.buffer = None
        self.n_samples_received = -1
        self.samples_missed_count = 0

    @abstractproperty
    def info(self) -> Dict[str, Any]:
        """
        Property to access information about the data stream.

        Returns:
            info (dict): the info dictionary
        """
        pass

    @abstractmethod
    def receive(self) -> np.ndarray:
        """
        Fetch new samples from the input stream.

        Returns:
            data (np.ndarray): an array with newly acquired data samples with shape (Channels, Time)
        """
        pass

    def update(self, data: Dict[str, Data]) -> int:
        """
        This function is called by the Manager to update the data dict with new data.

        Parameters:
            data (Dict[str, Data]): the data dict to update

        Returns:
            n_samples_received (int): the number of new samples received
        """
        if self.buffer is None:
            # initialize raw buffer
            buffer_size = int(self.info["sfreq"] * self.buffer_seconds)
            self.buffer = deque(maxlen=buffer_size)

        # fetch new data
        new_data = self.receive()
        if new_data is None:
            self.n_samples_received = -1
            return -1

        # rescale data
        new_data *= self.rescale

        # make sure we didn't receive more samples than the buffer can hold
        self.n_samples_received = new_data.shape[1]
        if self.n_samples_received > self.buffer.maxlen:
            self.n_samples_received = self.buffer.maxlen
            self.samples_missed_count += 1
            print(
                f"Received {self.n_samples_received} new samples but the buffer only holds "
                f"{self.buffer.maxlen} samples. Output modules will miss some samples. "
                f"({self.samples_missed_count})"
            )

        # update raw buffer
        self.buffer.extend(new_data.T)

        # skip processing and output steps while the buffer is not full
        if len(self.buffer) < self.buffer.maxlen:
            return -1

        # update data dict
        raw_buff = np.asarray(self.buffer)
        for i, ch in enumerate(self.info["ch_names"]):
            addr = fmt_address(f"/{self.address}/{ch}")
            # select single channel from raw data
            raw = RawData(raw_buff, self.info, i)
            # insert raw into data dict
            data[addr] = Data(addr, raw, DataType.RAW_CHANNEL)
        return self.n_samples_received


class DataOut(ABC):
    SUPPORTED_DTYPES = None
    """
    Abstract data output stream. Derive from this class to implement new output streams.
    """

    @abstractmethod
    def output(self, data: List[Data]):
        """
        Output the current data.

        Parameters:
            data (List[Data]): the data dict to send
        """
        pass

    def update(self, data: Dict[str, Data]):
        """
        Deriving classes should not override this method. It get's called by the Manager,
        selects which data to output and calls the abstract output method.

        Parameters:
            data (Dict[str, Data]): the data dict to send
        """
        if self.SUPPORTED_DTYPES is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not define SUPPORTED_DTYPES. All DataOut "
                "classes must define this class variable."
            )

        self.output([d for d in data.values() if d.dtype in self.SUPPORTED_DTYPES])


class Processor(ABC):
    SUPPORTED_DTYPES = None

    """
    Abstract data processor. Derive from this class to implement new feature extractors.

    Parameters:
        output_address (str): the address of the output stream
        input_addresses (str): the addresses of the input streams
        reduce (str): the reduction method to use, can be one of None, 'mean', 'median', 'max', 'min', 'std'
    """

    def __init__(
        self, output_address: str, *input_addresses: str, reduce: Optional[str] = None
    ):
        assert reduce in [
            None,
            "mean",
            "median",
            "max",
            "min",
            "std",
        ], f"Invalid reduce method '{reduce}'. Must be one of None, 'mean', 'median', 'max', 'min', 'std'."

        self.output_address = output_address
        self.input_addresses = input_addresses
        self.reduce = reduce

        if self.SUPPORTED_DTYPES is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not define SUPPORTED_DTYPES. All Processors "
                f"must define this class variable."
            )

    @abstractmethod
    def process(
        self, data: List[Data]
    ) -> Union[Data, List[Data], Dict[str, Data], Dict[str, List[Data]]]:
        """
        Process some input data and return the extracted features. The input data is a list of
        Data objects, filtered to match the SUPPORTED_DTYPES and input_addresses of this Processor.

        Note: all lists in the return value will be reduced to a single Data object if reduce is not None.

        Parameters:
            data (List[Data]): the data dict containing the input Data objects

        Returns:
            - features: the extracted features, can be one of
                - Data: a single Data object
                - List[Data]: a list of Data objects
                - Dict[str, Data]: a dictionary of Data objects
                - Dict[str, List[Data]]: a dictionary of lists of Data objects
        """
        pass

    def update(self, data: Dict[str, Data]):
        """
        Deriving classes should not override this method. It get's called by the Manager,
        applies channel selection and calles the process method with the channel subset.

        Parameters:
            data (Dict[str, Data]): the data dict containing the input Data objects
        """
        if self.SUPPORTED_DTYPES is None:
            raise RuntimeError(
                "Deriving classes must call the super().__init__() method in their constructor"
            )

        # select channels
        subset = [
            data[addr]
            for addr in expand_address(self.input_addresses, list(data.keys()))
            if data[addr].dtype in self.SUPPORTED_DTYPES
        ]
        # process the data
        result = self.process(subset)

        # insert processed data into data dict
        self.insert_result(result, data)

    def insert_result(
        self,
        result: Union[Data, List[Data], Dict[str, Data], Dict[str, List[Data]]],
        data: Dict[str, Data],
        suffix: str = "",
    ):
        """
        Insert the result of the process method into the data dict, applying the reduce method if necessary.
        """
        if isinstance(result, Data):
            self.update_address(result, suffix)
            data[result.address] = result
        elif isinstance(result, list):
            if self.reduce is None:
                # insert all items in the list
                for item in result:
                    self.insert_result(item, data, suffix=suffix)
            else:
                reduced_data = self.reduce_list(result, suffix)
                self.insert_result(reduced_data, data)
        elif isinstance(result, dict):
            for key, value in result.items():
                self.insert_result(value, data, suffix=f"{suffix}/{key}")
        else:
            raise TypeError(f"Unsupported result type: {type(result)}")

    def reduce_list(self, data_list: List[Data], suffix: str = "") -> Data:
        """
        Reduce a list of Data objects to a single Data object using the specified reduction method.
        """
        if self.reduce is None:
            raise ValueError(
                "Can't reduce a list of Data objects without a reduction method"
            )

        # make sure the dtype is reducible
        assert data_list[0].dtype in [
            DataType.FLOAT,
            DataType.ARRAY_1D,
            DataType.IMAGE,
        ], (
            f"Can't reduce a list of Data objects with dtype {data_list[0].dtype}. "
            "Supported dtypes are FLOAT, ARRAY_1D and IMAGE."
        )

        # make sure every item in the list has the same dtype
        assert all(
            data.dtype == data_list[0].dtype for data in data_list
        ), "Can't mix dtypes when reducing a list of Data objects"

        # split off the address' channel name
        addr = "/".join(data_list[0].address.split("/")[:-1])
        addr = f"{addr}/{suffix}/{self.reduce}"
        # prepare the data
        data_arr = np.stack([data.value for data in data_list], axis=0)
        dtype = data_list[0].dtype

        # reduce the list
        if self.reduce == "mean":
            return Data(addr, np.mean(data_arr, axis=0), dtype)
        elif self.reduce == "median":
            return Data(addr, np.median(data_arr, axis=0), dtype)
        elif self.reduce == "max":
            return Data(addr, np.max(data_arr, axis=0), dtype)
        elif self.reduce == "min":
            return Data(addr, np.min(data_arr, axis=0), dtype)
        elif self.reduce == "std":
            return Data(addr, np.std(data_arr, axis=0), dtype)
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduce}")

    def update_address(self, data: Data, suffix: Optional[str] = None):
        suffix = f"/{suffix}" or ""
        data.address = fmt_address(f"/{self.output_address}/{data.address}{suffix}")
        return data


class Normalization(ABC):
    """
    Abstract normalization class for implementing different normalization strategies.

    Designed to be used with the Processor class, it provides an interface for normalizing
    extracted features. The normalize method applies the strategy, and the reset method
    resets the normalization parameters to their initial state.
    """

    def __init__(self):
        self.user_input_thread = threading.Thread(
            target=self.reset_handler, daemon=True
        )
        self.user_input_thread.start()

    @abstractmethod
    def normalize(self, processed: Dict[str, float]):
        """
        This function is called by the manager to normalize the processed features according
        to the deriving class' normalization strategy. It should modify the processed dictionary
        in-place.

        Parameters:
            processed (Dict[str, float]): dictionary of extracted, unnormalized features
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This function should reset all running or statically acquired normalization parameters
        and reset the normalization to the initial state.
        """
        pass

    def reset_handler(self):
        """
        This function runs in a separate thread to wait for keyboard input, resetting the
        normalization state.
        """
        while True:
            input("Press enter to reset normalization parameters.\n")
            self.reset()


def fmt_address(address: str) -> str:
    """Remove repeated and trailing slashes from the address string and make sure it starts with a slash."""
    return "/" + re.sub(r"/+", "/", address).strip("/")


def expand_address(
    address: Union[str, Tuple[str], List[str]], available_addresses: List[str]
) -> List[str]:
    """
    Expand address patterns into an exhaustive list of addresses. Pattern matching follows
    regular expression syntax.

    Note that forward slashes will be escaped automatically. If you want to match a literal
    forward slash, you need to escape it with a backslash.

    Parameters:
        address (Union[str, Tuple[str], List[str]]): Address patterns, e.g. "/foo/.*".
        data (List[str]): List of addresses to match against.

    Returns:
        addresses (List[str]): A list of expanded addresses.
    """
    if isinstance(address, str):
        return expand_address((address,), available_addresses)
    elif len(address) == 0:
        # empty address list matches all
        return available_addresses

    # expand addresses
    if len(address) == 1:
        # escape forward slashes
        address = address[0].replace("/", r"\/")
        # match against all addresses
        return [a for a in available_addresses if re.match(address, a)]
    else:
        # recursively expand addresses
        return sum([expand_address(a, available_addresses) for a in address], [])


def viz_scale_colors(scale: List[float], fund: float) -> List[Tuple[int, int, int]]:
    """
    Convert a musical scale into a list of HSV colors based on the scale's frequency values
    and their averaged consonance.

    Parameters:
        scale (List[float]): A list of frequency ratios representing the musical scale.
        fund (float): The fundamental frequency of the scale in Hz.

    Returns:
        hsv_all (List[Tuple[float, float, float]]): A list of HSV color tuples, one for each scale step,
        with hue representing the frequency value, saturation representing the consonance, and
        luminance set to a fixed value.
    """

    min_ = 0
    max_ = 1
    # convert the scale to frequency values
    scale_freqs = scale2freqs(scale, fund)
    # compute the averaged consonance of each step
    scale_cons, _ = tuning_cons_matrix(scale, dyad_similarity, ratio_type="all")
    # rescale to match RGB standards (0, 255)
    scale_cons = (np.array(scale_cons) - min_) * (1 / max_ - min_) * 255
    scale_cons = scale_cons.astype("uint8").astype(float) / 255

    hsv_all = []
    for s, cons in zip(scale_freqs, scale_cons):
        # convert freq in nanometer values
        _, _, nm, octave = audible2visible(s)
        # convert to RGB values
        rgb = wavelength_to_rgb(nm)
        # convert to HSV values
        # TODO: colorsys might be slow
        hsv = colorsys.rgb_to_hsv(
            rgb[0] / float(255), rgb[1] / float(255), rgb[2] / float(255)
        )
        hsv = np.array(hsv)
        # rescale
        hsv = (hsv - 0) * (1 / (1 - 0))
        # define the saturation
        hsv[1] = cons
        # define the luminance
        hsv[2] = 200 / 255
        hsv = tuple(hsv)
        hsv_all.append(hsv)

    return hsv_all


def biotuner_realtime(data, Fs):
    bt_plant = compute_biotuner(peaks_function="harmonic_recurrence", sf=Fs)
    bt_plant.peaks_extraction(
        np.array(data),
        graph=False,
        min_freq=0.1,
        max_freq=65,
        precision=0.1,
        nIMFs=5,
        n_peaks=5,
        smooth_fft=4,
    )
    bt_plant.peaks_extension(method="harmonic_fit")
    bt_plant.compute_peaks_metrics(n_harm=3, delta_lim=50)
    harm_tuning = harmonic_tuning(bt_plant.all_harmonics)
    # bt_plant.compute_diss_curve(plot=True, input_type='peaks')
    # bt_plant.compute_spectromorph(comp_chords=True, graph=False)
    peaks = bt_plant.peaks
    extended_peaks = bt_plant.peaks
    metrics = bt_plant.peaks_metrics
    if not isinstance(metrics["subharm_tension"][0], float):
        metrics["subharm_tension"][0] = -1
    tuning = bt_plant.peaks_ratios
    return peaks, extended_peaks, metrics, tuning, harm_tuning


def bioelements_realtime(data, Fs):
    biotuning = compute_biotuner(
        1000,
        peaks_function="EMD",
        precision=0.1,
        n_harm=100,
        ratios_n_harms=10,
        ratios_inc_fit=False,
        ratios_inc=False,
    )  # Initialize biotuner object
    biotuning.peaks_extraction(data, ratios_extension=True, max_freq=50)
    _, _, _ = biotuning.peaks_extension(
        method="harmonic_fit", harm_function="mult", cons_limit=0.1
    )
    peaks_nm = [hertz_to_nm(x) for x in biotuning.extended_peaks]
    print("PEAKS BIOLEMENTS", peaks_nm)
    res = find_matching_spectral_lines(
        peaks_nm, convert_to_nm(ALL_ELEMENTS), tolerance=10
    )
    return res


# Helper function for computing a single connectivity matrix
def compute_conn_matrix_single(data, sf):
    bt_conn = harmonic_connectivity(
        sf=sf,
        data=data,
        peaks_function="harmonic_recurrence",
        precision=0.1,
        min_freq=2,
        max_freq=45,
        n_peaks=5,
    )
    bt_conn.compute_harm_connectivity(metric="harmsim", save=False, graph=False)
    return bt_conn.conn_matrix


def rgb2name(rgb):
    """
    Find the closest color in a dictionary of colors to an input RGB value.

    Parameters:
        rgb (Tuple[int, int, int]): RGB color tuple

    Returns:
        str: The name of the closest color in the dictionary
    """
    colors = {
        k: webcolors.hex_to_rgb(k) for k in webcolors.constants.CSS3_HEX_TO_NAMES.keys()
    }
    closest_color = min(
        colors,
        key=lambda color: sum((a - b) ** 2 for a, b in zip(rgb, colors[color])),
    )
    return webcolors.constants.CSS3_HEX_TO_NAMES[closest_color]


def text2speech(txt, lang="en"):
    gTTS(text=txt, lang=lang).write_to_fp(voice := NamedTemporaryFile())
    playsound(voice.name)
    voice.close()
