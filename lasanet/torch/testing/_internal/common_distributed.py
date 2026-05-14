import faulthandler
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from datetime import timedelta
from enum import Enum
from functools import partial, reduce
from functools import wraps
from io import StringIO
from typing import NamedTuple, Optional, Union

import torch
import torch.cuda.nccl
import torch.distributed as c10d
from torch.testing._internal.common_utils import (
    TestCase,
    TEST_WITH_ROCM,
    FILE_SCHEMA,
    find_free_port,
    retry_on_connect_failures,
    IS_SANDCASTLE,
    sandcastle_skip_if,
    sandcastle_skip,
)

logger = logging.getLogger(__name__)

class TestSkip(NamedTuple):
    exit_code: int
    message: str

TEST_SKIPS = {
    "backend_unavailable": TestSkip(
        72, "Skipped because distributed backend is not available."
    ),
    "small_worldsize": TestSkip(73, "Skipped due to small world size."),
    "no_cuda": TestSkip(74, "CUDA is not available."),
    "multi-gpu-1": TestSkip(75, "Need at least 1 CUDA device"),
    "multi-gpu-2": TestSkip(77, "Need at least 2 CUDA devices"),
    "multi-gpu-3": TestSkip(80, "Need at least 3 CUDA devices"),
    "multi-gpu-4": TestSkip(81, "Need at least 4 CUDA devices"),
    "multi-gpu-5": TestSkip(82, "Need at least 5 CUDA devices"),
    "multi-gpu-6": TestSkip(83, "Need at least 6 CUDA devices"),
    "multi-gpu-7": TestSkip(84, "Need at least 7 CUDA devices"),
    "multi-gpu-8": TestSkip(85, "Need at least 8 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm"),
    "no_peer_access": TestSkip(79, "Test skipped because no GPU peer access"),
    "generic": TestSkip(
        86, "Test skipped at subprocess level, look at subprocess log for skip reason"
    ),
}

def skip_if_no_gpu(func):
    """Nccl multigpu tests require at least 2 GPUS. Skip if this is not met"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)

        return func(*args, **kwargs)

    return wrapper

def skip_if_small_worldsize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) <= 2:
            sys.exit(TEST_SKIPS["small_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper

def require_n_gpus_for_nccl_backend(n, backend):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend == "nccl" and torch.cuda.device_count() < n:
                sys.exit(TEST_SKIPS[f"multi-gpu-{n}"].exit_code)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator

def skip_if_lt_x_gpu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)

        return wrapper

    return decorator

def nccl_skip_if_lt_x_gpu(backend, x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend != "nccl":
                return func(*args, **kwargs)
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)

        return wrapper

    return decorator

def verify_ddp_error_logged(model_DDP, err_substr):

    ddp_logging_data = model_DDP._get_ddp_logging_data()
    assert "has_error" in ddp_logging_data
    assert "error" in ddp_logging_data
    assert err_substr in ddp_logging_data["error"]

def with_nccl_blocking_wait(func):
    """
    Convenience decorator to set/unset NCCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of NCCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both NCCL_BLOCKING_WAIT and
    NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        try:
            cached_nccl_async_error_handling: Union[str, None] = os.environ[
                "NCCL_ASYNC_ERROR_HANDLING"
            ]
            del os.environ["NCCL_ASYNC_ERROR_HANDLING"]
        except KeyError:

            cached_nccl_async_error_handling = None

        try:
            cached_nccl_blocking_wait: Union[str, None] = os.environ[
                "NCCL_BLOCKING_WAIT"
            ]
        except KeyError:
            cached_nccl_blocking_wait = None
        finally:
            os.environ["NCCL_BLOCKING_WAIT"] = "1"

        try:
            ret = func(*args, **kwargs)
            return ret
        finally:

            if cached_nccl_async_error_handling is not None:
                os.environ[
                    "NCCL_ASYNC_ERROR_HANDLING"
                ] = cached_nccl_async_error_handling

            if cached_nccl_blocking_wait is not None:
                os.environ["NCCL_BLOCKING_WAIT"] = cached_nccl_blocking_wait

    return wrapper

def with_dist_debug_levels(levels):
    """
    Runs a test for each distributed debug level specified in levels.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_level = os.environ.get("TORCH_DISTRIBUTED_DEBUG", None)
            for level in levels:
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = level
                ret = func(*args, **kwargs)
                if old_level is not None:
                    os.environ["TORCH_DISTRIBUTED_DEBUG"] = old_level

            return ret

        return wrapper

    return decorator

def requires_gloo():
    return sandcastle_skip_if(
        not c10d.is_gloo_available(),
        "c10d was not compiled with the Gloo backend",
    )

def requires_nccl_version(version, msg):
    if not c10d.is_nccl_available():
        return sandcastle_skip(
            "c10d was not compiled with the NCCL backend",
        )
    else:
        return sandcastle_skip_if(
            torch.cuda.nccl.version() < version,
            "Requires NCCL version greater than or equal to: {}, found: {}, reason: {}".format(
                version, torch.cuda.nccl.version(), msg
            ),
        )

def requires_nccl():
    return sandcastle_skip_if(
        not c10d.is_nccl_available(),
        "c10d was not compiled with the NCCL backend",
    )

def requires_mpi():
    return sandcastle_skip_if(
        not c10d.is_mpi_available(),
        "c10d was not compiled with the MPI backend",
    )

def skip_if_rocm(func):
    """Skips a test for ROCm"""
    func.skip_if_rocm = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS["skipIfRocm"].exit_code)

    return wrapper

def skip_if_win32():
    return sandcastle_skip_if(
        sys.platform == 'win32',
        "This unit test case is not supportted on Windows platform",
    )

@retry_on_connect_failures
def create_tcp_store(
    addr="localhost",
    world_size=1,
    is_master=True,
    timeout=timedelta(minutes=5),
    wait_for_workers=True,
    jit_class=False,
):
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    port = find_free_port()
    if jit_class:
        timeout_millisecond = int(timeout / timedelta(milliseconds=1))
        return torch.classes.dist_c10d.TCPStore(
            addr, port, world_size, is_master, timeout_millisecond
        )
    else:
        return c10d.TCPStore(
            addr, port, world_size, is_master, wait_for_workers=wait_for_workers
        )

TIMEOUT_DEFAULT = 100
TIMEOUT_OVERRIDE = {"test_ddp_uneven_inputs": 400}

def create_device(interface=None):
    if sys.platform == "win32" or interface is None:
        return c10d.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    else:
        return c10d.ProcessGroupGloo.create_device(interface=interface)

def get_timeout(test_id) -> int:
    return TIMEOUT_OVERRIDE.get(test_id.split(".")[-1], TIMEOUT_DEFAULT)

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def simple_sparse_reduce_tests(rank: int, world_size: int, num_inputs: int = 1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """

    def generate(rank: int, world_size: int, sparse_dims: int = 1, dense_dims: int = 0):

        indices = torch.reshape(torch.arange(rank + 1), (1, rank + 1))
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices = torch.cat((indices, torch.zeros(1, rank + 1)))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size: int):
        return reduce(
            lambda a, b: a + b, [fn(rank, world_size) for rank in range(world_size)]
        )

    return [
        (
            [
                fn(num_inputs * rank + i, num_inputs * world_size)
                for i in range(num_inputs)
            ],
            [compute_sum(fn, num_inputs * world_size) for i in range(num_inputs)],
        )
        for fn in [
            partial(generate, sparse_dims=1),
            partial(generate, sparse_dims=2),
            partial(generate, sparse_dims=3),
            partial(generate, dense_dims=1),
            partial(generate, dense_dims=2),
            partial(generate, dense_dims=3),
        ]
    ]

tmp_dir: Optional[tempfile.TemporaryDirectory] = None

def initialize_temp_directories(init_method: Optional[str] = None) -> None:
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["TEMP_DIR"] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, "barrier"))
    os.mkdir(os.path.join(tmp_dir.name, "test_dir"))
    init_dir_path = os.path.join(tmp_dir.name, "init_dir")
    os.mkdir(init_dir_path)

    if init_method is not None:
        os.environ["INIT_METHOD"] = init_method
    else:
        os.environ["INIT_METHOD"] = FILE_SCHEMA + os.path.join(
            init_dir_path, "shared_init_file"
        )

def cleanup_temp_dir() -> None:
    if tmp_dir is not None:
        tmp_dir.cleanup()

class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1

    TEST_ERROR_EXIT_CODE = 10

    def _should_stop_test_suite(self) -> bool:
        return False

    @property
    def world_size(self) -> int:
        return 4

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name

        self.pid_to_pipe = {}

    def tearDown(self) -> None:
        super().tearDown()
        for p in self.processes:
            p.terminate()

        self.processes = []

    def _current_test_name(self) -> str:

        return self.id().split(".")[-1]

    def _start_processes(self, proc) -> None:
        self.processes = []
        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name(), self.file_name, child_conn),
            )
            process.start()
            logger.info(f"Started process {rank} with pid {process.pid}")
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.info(f"Starting event listener thread for {rank}")
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            if parent_pipe in ready_pipes:

                if parent_pipe.closed:
                    logger.info(
                        f"Pipe closed for process {rank}, stopping event listener thread"
                    )
                    return

                event = parent_pipe.recv()
                logger.info(f"Received event {event} on process {rank}")

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:

                    with tempfile.NamedTemporaryFile(mode="r+") as tmp_file:
                        faulthandler.dump_traceback(tmp_file)

                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())

                        logger.info(f"Process {rank} sent traceback")

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        self = cls(test_name)

        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, rank),
            daemon=True,
        )
        event_listener_thread.start()

        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe, signal_send_pipe, event_listener_thread)

    def run_test(
        self, test_name: str, parent_pipe, signal_pipe=None, event_listener_thread=None
    ) -> None:
        if sys.platform != "win32" and sys.platform != "darwin":

            torch._C._set_print_stack_traces_on_fatal_signal(True)

        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(
                f"Process {self.rank} skipping test {test_name} for following reason: {str(se)}"
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
        except Exception as e:
            logger.error(
                f"Caught exception: \n{traceback.format_exc()} exiting "
                f"process {self.rank} with exit code: {MultiProcessTestCase.TEST_ERROR_EXIT_CODE}"
            )

            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_pipe is not None:
                signal_pipe.send(None)
            if event_listener_thread is not None:
                event_listener_thread.join()

            parent_pipe.close()

    def _get_timedout_process_traceback(self) -> None:
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error(
                        f"Encountered error while trying to get traceback for process {i}: {e}"
                    )

        for rank, pipe in pipes:
            try:

                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(
                            f"Pipe closed for process {rank}, cannot retrieve traceback"
                        )
                        continue

                    traceback = pipe.recv()
                    logger.error(
                        f"Process {rank} timed out with traceback: \n\n{traceback}"
                    )
                else:
                    logger.error(
                        f"Could not retrieve traceback for timed out process: {rank}"
                    )
            except ConnectionError as e:
                logger.error(
                    f"Encountered error while trying to get traceback for process {rank}: {e}"
                )

    def _join_processes(self, fn) -> None:
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:

                for (i, p) in enumerate(self.processes):

                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(
                            f"Process {i} terminated with exit code {p.exitcode}, terminating remaining processes."
                        )
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break

                if all([p.exitcode is not None for p in self.processes]):
                    break

                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    print(
                        f"Timing out after {timeout} seconds and killing subprocesses."
                    )
                    for p in self.processes:
                        p.terminate()
                    break

                time.sleep(0.1)

            elapsed_time = time.time() - start_time

            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:

            for pid, pipe in self.pid_to_pipe.items():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    "Process {} timed out after {} seconds".format(i, elapsed_time)
                )
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]

        errored_processes = [
            (i, p)
            for i, p in enumerate(self.processes)
            if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]
        if errored_processes:
            error = ""
            for i, process in errored_processes:

                error_message = self.pid_to_pipe[process.pid].recv()
                error += (
                    "Process {} exited with error code {} and exception:\n{}\n".format(
                        i, MultiProcessTestCase.TEST_ERROR_EXIT_CODE, error_message
                    )
                )

            raise RuntimeError(error)

        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    "Process {} terminated or timed out after {} seconds".format(
                        i, elapsed_time
                    )
                )
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg="Expect process {} exit code to match Process 0 exit code of {}, but got {}".format(
                    i, first_process.exitcode, p.exitcode
                ),
            )
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                if IS_SANDCASTLE:

                    logger.info(
                        f"Skipping {self.id()} on sandcastle for the following reason: {skip.message}"
                    )
                    return
                else:
                    raise unittest.SkipTest(skip.message)
        self.assertEqual(
            first_process.exitcode,
            0,
            msg="Expected zero exit code but got {} for pid: {}".format(first_process.exitcode, first_process.pid)
        )

    @property
    def is_master(self) -> bool:
        return self.rank == 0
