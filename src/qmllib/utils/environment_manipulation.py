import contextlib
import ctypes


def mkl_set_num_threads(cores: int) -> None:

    try:
        mkl_rt = ctypes.CDLL("libmkl_rt.so")
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
    except OSError:
        pass


def mkl_get_num_threads():

    try:
        mkl_rt = ctypes.CDLL("libmkl_rt.so")
        mkl_num_threads = mkl_rt.mkl_get_max_threads()
        return mkl_num_threads

    except OSError:

        return None


def mkl_reset_num_threads():

    try:
        mkl_rt = ctypes.CDLL("libmkl_rt.so")
        mkl_num_threads = mkl_rt.mkl_get_max_threads()
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(mkl_num_threads)))

    except OSError:
        pass


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """Temporarily updates the environment dictionary in-place."""
    # TODO Use context manager for thread cores changes
    raise NotImplementedError()
