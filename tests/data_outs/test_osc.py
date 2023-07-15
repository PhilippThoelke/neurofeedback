import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from neurofeedback.data_out import OSCStream
from neurofeedback.utils import Data, DataType


def split_dtypes(combined: DataType) -> List[DataType]:
    components = []
    for dtype in DataType:
        if dtype in combined:
            components.append(dtype)
    return components


def build_data(addr: str, dtype: DataType) -> Data:
    if dtype == DataType.FLOAT:
        return Data(addr, 0.5, dtype)
    elif dtype == DataType.STRING:
        return Data(addr, "hello", dtype)
    elif dtype == DataType.ARRAY_1D:
        return Data(addr, np.arange(3, dtype=float), dtype)
    elif dtype == DataType.IMAGE:
        return Data(addr, np.random.rand(10, 10, 3), dtype)
    else:
        raise ValueError(f"Unsupported data type {dtype}")


def setup_server(address: str) -> Tuple[ThreadingOSCUDPServer, Dict[str, Any]]:
    result = dict()

    def receive_msg(addr, val):
        result[addr] = val

    # define a message handler
    disp = Dispatcher()
    disp.map(address, receive_msg)

    # find an open port and instantiate the server
    for port in range(5005, 5025):
        try:
            server = ThreadingOSCUDPServer(("127.0.0.1", port), disp)
            break
        except OSError:
            print(f"Port {port} is already in use")

    # start listening in a separate thread
    def server_thread():
        server.serve_forever()

    server_thread = threading.Thread(target=server_thread, daemon=True)
    server_thread.start()
    return server, result


def receive_timeout(msg: Data, result: Dict[str, Any], timeout: float = 2):
    wait_start = time.time()
    while msg.address not in result:
        if time.time() - wait_start > timeout:
            raise TimeoutError(f"Didn't receive message after {timeout} seconds")
        time.sleep(0.1)


@pytest.mark.parametrize("dtype", split_dtypes(OSCStream.SUPPORTED_DTYPES))
def test_send_receive(dtype: DataType, address: str = "/test"):
    # setup server
    server, result = setup_server(address)

    # send data object
    msg = build_data(address, dtype)
    osc = OSCStream("127.0.0.1", server.server_address[1])
    osc.update({msg.address: msg})

    # wait for the message to be received
    receive_timeout(msg, result)

    # check that the message was received correctly
    if dtype == DataType.STRING:
        assert (
            result[msg.address] == msg.value
        ), "Received message did not match sent message"
    else:
        np.testing.assert_allclose(
            result[msg.address], msg.value
        ), "Received message did not match sent message"

    server.shutdown()


@pytest.mark.skip("Large messages are currently broken")
@pytest.mark.parametrize("resolution", [32, 128, 512])
def test_large_message(resolution: int, address: str = "/test"):
    # setup server
    server, result = setup_server(address)

    # send data object
    msg = Data(address, np.random.rand(resolution, resolution, 3), DataType.IMAGE)
    osc = OSCStream("127.0.0.1", server.server_address[1])
    osc.update({msg.address: msg})

    # wait for the message to be received
    receive_timeout(msg, result)

    # check that the message was received correctly
    np.testing.assert_allclose(
        result[msg.address], msg.value
    ), "Received message did not match sent message"

    server.shutdown()
