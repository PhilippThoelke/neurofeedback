from neurofeedback import data_out, manager, normalization, processors
from tests.utils import DummyStream


def test_manager():
    mngr = manager.Manager(
        {"dummy": DummyStream()},
        [processors.LempelZiv()],
        normalization.WelfordsZTransform(),
        [data_out.OSCStream("127.0.0.1", 5005)],
    )
    mngr.run(n_iterations=10)
