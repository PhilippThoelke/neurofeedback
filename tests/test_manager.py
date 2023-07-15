from neurofeedback import data_out, manager, normalization, processors
from tests.utils import DummyStream


def test_manager():
    mngr = manager.Manager(
        DummyStream(),
        [processors.LempelZiv()],
        normalization.WelfordsZTransform(),
        [data_out.OSCStream("127.0.0.1", 5005)],
    )

    def callback(mngr: manager.Manager, it: int):
        assert it < 10, "Callback called too many times"

        for dat in mngr.data.values():
            assert dat.dirty, "Data not marked as dirty after update"

    mngr.run(n_iterations=10, callback=callback)
    assert len(mngr.data) != 0, "Manager cleared the data dictionary"
