import time
from typing import Callable, List, Union

from neurofeedback.utils import DataIn, DataOut, Normalization, Processor


class Manager:
    """
    Central class to manage an EEG input stream, several processing steps,
    feature normalization and multiple output channels.

    Parameters:
        data_in (List[DataIn]): a list of DataIn channels (e.g. EEGStream, EEGRecording)
        processors (List[Processor]): a list of Processor instances (e.g. PSD, LempelZiv)
        normalization (Normalization): the normalization strategy to apply to the extracted features
        data_out (List[DataOut]): a list of DataOut channels (e.g. OSCStream, PlotProcessed)
        frequency (int): frequency of the data processing loop (-1 to run as fast as possible)
    """

    def __init__(
        self,
        data_in: Union[DataIn, List[DataIn]],
        processors: List[Processor],
        normalization: Normalization,
        data_out: List[DataOut],
        frequency: int = 10,
    ):
        # check for duplicate processor labels
        labels = []
        for processor in processors:
            if processor.output_suffix in labels:
                raise ValueError(
                    f"Duplicate processor label '{processor.output_suffix}' detected"
                )
            labels.append(processor.output_suffix)

        # assign pipeline components
        self.data_in = data_in if isinstance(data_in, list) else [data_in]
        self.processors = processors
        self.normalization = normalization
        self.data_out = data_out

        # auxiliary attributes
        self.frequency = frequency
        self.too_slow_count = 0
        self.filling_buffer = True
        self.data = dict()

    def update(self):
        """
        Fetches new data from the input channels, processes the data and
        outputs the processed data to the output channels.
        """
        # fetch raw data
        if any(d.update(self.data) == -1 for d in self.data_in):
            return
        elif self.filling_buffer:
            print("done")
            self.filling_buffer = False

        # process raw data (feature extraction)
        for processor in self.processors:
            processor.update(self.data)

        # update data outputs
        for out in self.data_out:
            out.update(self.data)

        # set data from last iteration to dirty
        for key in self.data:
            self.data[key].dirty = True

    def run(self, n_iterations: int = -1, callback: Callable = None):
        """
        Start the fetching and processing loop and limit the loop to run at a
        constant update rate.

        Parameters:
            n_iterations (int): number of iterations to run the loop for (-1 for infinite)
            callback (Callable): iteration callback with signature callback(mngr, it)
        """
        print("Filling buffer(s)...", end="")

        last_time = time.time()
        it = 0
        while it != n_iterations:
            # receive, process and output data
            self.update()

            # call callback function
            if callback is not None:
                callback(self, it)
            it += 1

            # limit the loop to run at a constant update rate
            if self.frequency > 0:
                # ensure a constant sampling frequency
                current_time = time.time()
                sleep_dur = 1 / self.frequency - current_time + last_time
                if sleep_dur >= 0:
                    time.sleep(sleep_dur)
                else:
                    self.too_slow_count += 1
                    print(
                        f"Processing too slow to run at {self.frequency}Hz"
                        f" ({self.too_slow_count})"
                    )
                last_time = time.time()


if __name__ == "__main__":
    from neurofeedback import data_in, data_out, normalization, processors

    mngr = Manager(
        data_in=[
            data_in.EEGRecording.make_eegbci(),
            # "file": data_in.EEGStream("Muse00:55:DA:B0:49:D3"),
        ],
        processors=[
            processors.PSD(band="delta"),
            # processors.PSD(label="theta"),
            # processors.PSD(label="alpha"),
            # processors.PSD(label="beta"),
            # processors.PSD(label="gamma"),
            processors.LempelZiv("/eegbci/F.*", reduce=None),
            # processors.Ratio("/file/alpha", "/file/theta", "alpha/theta"),
            # processors.Bioelements(channels={"file": ["AF7"]}),
            # processors.Biocolor(channels={"file": ["AF7"]}),
            # processors.OpenAI(
            #     "/file/biocolor/ch0_peak0_name",
            #     "/file/bioelements/ch0_bioelements",
            #     read_text=True,
            # ),
            # processors.Biotuner(channels={"file": ["O1", "O2"]}),
        ],
        normalization=normalization.StaticBaselineNormal(duration=30),
        data_out=[
            data_out.OSCStream("127.0.0.1", 5005),
            # data_out.PlotRaw("file"),
            # data_out.PlotProcessed(),
        ],
    )

    mngr.run()
