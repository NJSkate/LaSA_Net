import copy
import gc
import inspect
import runpy
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps
from typing import List, Any, ClassVar, Optional, Sequence, Tuple
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL,\
    skipCUDANonDefaultStreamIf, TEST_WITH_ASAN, TEST_WITH_UBSAN, TEST_WITH_TSAN,\
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, DeterministicGuard, TEST_SKIP_NOARCH,\
    _TestParametrizer, dtype_name, TEST_WITH_MIOPEN_SUGGEST_NHWC
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_dtype import get_all_dtypes

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def _dtype_test_suffix(dtypes):
    """ Returns the test suffix for a dtype, sequence of dtypes, or None. """
    if isinstance(dtypes, list) or isinstance(dtypes, tuple):
        if len(dtypes) == 0:
            return ''
        return '_' + '_'.join((dtype_name(d) for d in dtypes))
    elif dtypes:
        return '_{}'.format(dtype_name(dtypes))
    else:
        return ''

def _update_param_kwargs(param_kwargs, name, value):
    """ Adds a kwarg with the specified name and value to the param_kwargs dict. """
    if isinstance(value, list) or isinstance(value, tuple):

        param_kwargs['{}s'.format(name)] = value
    elif value:
        param_kwargs[name] = value

class DeviceTypeTestBase(TestCase):
    device_type: str = 'generic_device_type'

    _stop_test_suite = False

    _tls = threading.local()
    _tls.precision = TestCase._precision
    _tls.rel_tol = TestCase._rel_tol

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

    @property
    def rel_tol(self):
        return self._tls.rel_tol

    @rel_tol.setter
    def rel_tol(self, prec):
        self._tls.rel_tol = prec

    @classmethod
    def get_primary_device(cls):
        return cls.device_type

    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        return test.dtypes.get(cls.device_type, test.dtypes.get('all', None))

    def _get_precision_override(self, test, dtype):
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    def _get_tolerance_override(self, test, dtype):
        if not hasattr(test, 'tolerance_overrides'):
            return self.precision, self.rel_tol
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    def _apply_precision_override_for_test(self, test, param_kwargs):
        dtype = param_kwargs['dtype'] if 'dtype' in param_kwargs else None
        dtype = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else dtype
        if dtype:
            self.precision = self._get_precision_override(test, dtype)
            self.precision, self.rel_tol = self._get_tolerance_override(test, dtype)

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):

        def instantiate_test_helper(cls, name, *, test, param_kwargs=None):

            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):

                param_kwargs = {} if param_kwargs is None else param_kwargs
                test_sig_params = inspect.signature(test).parameters
                if 'device' in test_sig_params or 'devices' in test_sig_params:
                    device_arg: str = cls.get_primary_device()
                    if hasattr(test, 'num_required_devices'):
                        device_arg = cls.get_all_devices()
                    _update_param_kwargs(param_kwargs, 'device', device_arg)

                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:

                    self._stop_test_suite = self._should_stop_test_suite()

                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol

                return result

            assert not hasattr(cls, name), "Redefinition of test {0}".format(name)
            setattr(cls, name, instantiated_test)

        def default_parametrize_fn(test, generic_cls, cls):

            test_suffix = cls.device_type
            yield (test, test_suffix, {})

        parametrize_fn = test.parametrize_fn if hasattr(test, 'parametrize_fn') else default_parametrize_fn
        for (test, test_suffix, param_kwargs) in parametrize_fn(test, generic_cls, cls):
            if hasattr(test, 'handles_dtypes') and test.handles_dtypes:
                full_name = '{}_{}'.format(name, test_suffix)
                instantiate_test_helper(cls=cls, name=full_name, test=test, param_kwargs=param_kwargs)
            else:

                dtypes = cls._get_dtypes(test)
                dtypes = tuple(dtypes) if dtypes is not None else (None,)
                for dtype in dtypes:
                    all_param_kwargs = dict(param_kwargs)
                    _update_param_kwargs(all_param_kwargs, 'dtype', dtype)
                    full_name = '{}_{}{}'.format(name, test_suffix, _dtype_test_suffix(dtype))
                    instantiate_test_helper(cls=cls, name=full_name, test=test, param_kwargs=all_param_kwargs)

    def run(self, result=None):
        super().run(result=result)

        if self._stop_test_suite:
            result.stop()

class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'

    def _should_stop_test_suite(self):
        return False

class MetaTestBase(DeviceTypeTestBase):
    device_type = 'meta'
    _ignore_not_implemented_error = True

    def _should_stop_test_suite(self):
        return False

class CUDATestBase(DeviceTypeTestBase):
    device_type = 'cuda'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    def has_cudnn(self):
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = 'cuda:{0}'
        non_primary_devices = [cuda_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):

        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma

        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        cls.primary_device = 'cuda:{0}'.format(torch.cuda.current_device())

def get_device_type_test_bases():

    test_bases: List[Any] = list()

    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:

            if not TEST_WITH_ASAN and not TEST_WITH_TSAN and not TEST_WITH_UBSAN:
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
            test_bases.append(MetaTestBase)
    else:
        test_bases.append(CPUTestBase)
        if not TEST_SKIP_NOARCH:
            test_bases.append(MetaTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)

    return test_bases

device_type_test_bases = get_device_type_test_bases()

def filter_desired_device_types(device_type_test_bases, except_for=None, only_for=None):

    intersect = set(except_for if except_for else []) & set(only_for if only_for else [])
    assert not intersect, f"device ({intersect}) appeared in both except_for and only_for"

    if except_for:
        device_type_test_bases = filter(
            lambda x: x.device_type not in except_for, device_type_test_bases)
    if only_for:
        device_type_test_bases = filter(
            lambda x: x.device_type in only_for, device_type_test_bases)

    return list(device_type_test_bases)

_TORCH_TEST_DEVICES = os.environ.get('TORCH_TEST_DEVICES', None)
if _TORCH_TEST_DEVICES:
    for path in _TORCH_TEST_DEVICES.split(':'):

        mod = runpy.run_path(path, init_globals=globals())
        device_type_test_bases.append(mod['TEST_CLASS'])

PYTORCH_CUDA_MEMCHECK = os.getenv('PYTORCH_CUDA_MEMCHECK', '0') == '1'

PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY = 'PYTORCH_TESTING_DEVICE_ONLY_FOR'
PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY = 'PYTORCH_TESTING_DEVICE_EXCEPT_FOR'

def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None):

    del scope[generic_test_class.__name__]

    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, generic_test_class.__bases__, {})

    generic_members = set(generic_test_class.__dict__.keys()) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]

    desired_device_type_test_bases = filter_desired_device_types(device_type_test_bases,
                                                                 except_for, only_for)

    def split_if_not_empty(x: str):
        return x.split(",") if len(x) != 0 else []

    env_only_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    env_except_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))

    desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases,
                                                                 env_except_for, env_only_for)

    for base in desired_device_type_test_bases:

        if TEST_WITH_ROCM and base.device_type != 'cuda':
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()

        device_type_test_class: Any = type(class_name, (base, empty_class), {})

        for name in generic_members:
            if name in generic_tests:
                test = getattr(generic_test_class, name)

                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:

                    device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            else:
                assert name not in device_type_test_class.__dict__, "Redefinition of directly defined member {0}".format(name)
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)

        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class

class OpDTypes(Enum):
    basic = 0
    supported = 1
    unsupported = 2
    supported_backward = 3
    unsupported_backward = 4
    none = 5

class ops(_TestParametrizer):
    def __init__(self, op_list, *, dtypes: OpDTypes = OpDTypes.basic,
                 allowed_dtypes: Optional[Sequence[torch.dtype]] = None):
        super().__init__(handles_dtypes=True)
        self.op_list = op_list
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None

    def _parametrize_test(self, test, generic_cls, device_cls):
        """ Parameterizes the given test function across each op and its associated dtypes. """
        for op in self.op_list:

            dtypes = device_cls._get_dtypes(test)
            if dtypes is None:
                if self.opinfo_dtypes == OpDTypes.unsupported_backward:
                    dtypes = set(get_all_dtypes()).difference(op.supported_backward_dtypes(device_cls.device_type))
                elif self.opinfo_dtypes == OpDTypes.supported_backward:
                    dtypes = op.supported_backward_dtypes(device_cls.device_type)
                elif self.opinfo_dtypes == OpDTypes.unsupported:
                    dtypes = set(get_all_dtypes()).difference(op.supported_dtypes(device_cls.device_type))
                elif self.opinfo_dtypes == OpDTypes.supported:
                    dtypes = op.supported_dtypes(device_cls.device_type)
                elif self.opinfo_dtypes == OpDTypes.basic:
                    dtypes = op.default_test_dtypes(device_cls.device_type)
                elif self.opinfo_dtypes == OpDTypes.none:
                    dtypes = [None]
                else:
                    raise RuntimeError(f"Unknown OpDType: {self.opinfo_dtypes}")

                if self.allowed_dtypes is not None:
                    dtypes = dtypes.intersection(self.allowed_dtypes)
            else:
                assert self.allowed_dtypes is None, "ops(allowed_dtypes=[...]) and the dtypes decorator are incompatible"
                assert self.opinfo_dtypes == OpDTypes.basic, "ops(dtypes=...) and the dtypes decorator are incompatible"

            for dtype in dtypes:

                test_name = '{}{}_{}{}'.format(op.name.replace('.', '_'),
                                               '_' + op.variant_test_name if op.variant_test_name else '',
                                               device_cls.device_type,
                                               _dtype_test_suffix(dtype))

                param_kwargs = {'op': op}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)

                try:
                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    for decorator in op.get_decorators(
                            generic_cls.__name__, test.__name__, device_cls.device_type, dtype):
                        test_wrapper = decorator(test_wrapper)

                    yield (test_wrapper, test_name, param_kwargs)
                except Exception as ex:

                    print("Failed to instantiate {0} for op {1}!".format(test_name, op.name))
                    raise ex

class skipIf(object):

    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def dep_fn(slf, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (isinstance(self.dep, bool) and self.dep):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, *args, **kwargs)
        return dep_fn

class skipCPUIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='cpu')

class skipCUDAIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='cuda')

class skipMetaIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='meta')

def _has_sufficient_memory(device, size):
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            return False
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device) >= size

    if device == 'xla':
        raise unittest.SkipTest('TODO: Memory availability checks for XLA?')

    if device != 'cpu':
        raise unittest.SkipTest('Unknown device type')

    if not HAS_PSUTIL:
        raise unittest.SkipTest('Need psutil to determine if memory is sufficient')

    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size

    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size

def largeTensorTest(size, device=None):
    """Skip test if the device has insufficient memory to run the test

    size may be a number of bytes, a string of the form "N GB", or a callable

    If the test is a device generic test, available memory on the primary device will be checked.
    It can also be overriden by the optional `device=` argument.
    In other tests, the `device=` argument needs to be specified.
    """
    if isinstance(size, str):
        assert size.endswith("GB") or size.endswith("gb"), "only bytes or GB supported"
        size = 1024 ** 3 * int(size[:-2])

    def inner(fn):
        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            size_bytes = size(self, *args, **kwargs) if callable(size) else size
            _device = device if device is not None else self.get_primary_device()
            if not _has_sufficient_memory(_device, size_bytes):
                raise unittest.SkipTest('Insufficient {} memory'.format(_device))

            return fn(self, *args, **kwargs)
        return dep_fn
    return inner

class expectedFailure(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def efail_fn(slf, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                try:
                    fn(slf, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail('expected test to fail, but it passed')

            return fn(slf, *args, **kwargs)
        return efail_fn

class onlyOn(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = "Only runs on {0}".format(self.device_type)
                raise unittest.SkipTest(reason)

            return fn(slf, *args, **kwargs)

        return only_fn

class deviceCountAtLeast(object):

    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        assert not hasattr(fn, 'num_required_devices'), "deviceCountAtLeast redefinition for {0}".format(fn.__name__)
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if len(devices) < self.num_required_devices:
                reason = "fewer than {0} devices detected".format(self.num_required_devices)
                raise unittest.SkipTest(reason)

            return fn(slf, devices, *args, **kwargs)

        return multi_fn

def onlyOnCPUAndCUDA(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type != 'cpu' and self.device_type != 'cuda':
            reason = "onlyOnCPUAndCUDA: doesn't run on {0}".format(self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn

class precisionOverride(object):

    def __init__(self, d):
        assert isinstance(d, dict), "precisionOverride not given a dtype : precision dict!"
        for dtype, prec in d.items():
            assert isinstance(dtype, torch.dtype), "precisionOverride given unknown dtype {0}".format(dtype)

        self.d = d

    def __call__(self, fn):
        fn.precision_overrides = self.d
        return fn

tol = namedtuple('tol', ['atol', 'rtol'])

class toleranceOverride(object):
    def __init__(self, d):
        assert isinstance(d, dict), "toleranceOverride not given a dtype : tol dict!"
        for dtype, prec in d.items():
            assert isinstance(dtype, torch.dtype), "toleranceOverride given unknown dtype {0}".format(dtype)
            assert isinstance(prec, tol), "toleranceOverride not given a dtype : tol dict!"

        self.d = d

    def __call__(self, fn):
        fn.tolerance_overrides = self.d
        return fn

class dtypes(object):

    def __init__(self, *args, device_type="all"):
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)),\
                    "When one dtype variant is a tuple or list, "\
                    "all dtype variants must be. "\
                    "Received non-list non-tuple dtype {0}".format(str(arg))
                assert all(isinstance(dtype, torch.dtype) for dtype in arg), "Unknown dtype in {0}".format(str(arg))
        else:
            assert all(isinstance(arg, torch.dtype) for arg in args), "Unknown dtype in {0}".format(str(args))

        self.args = args
        self.device_type = device_type

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, "dtypes redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn

class dtypesIfCPU(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='cpu')

class dtypesIfCUDA(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='cuda')

def onlyCPU(fn):
    return onlyOn('cpu')(fn)

def onlyCUDA(fn):
    return onlyOn('cuda')(fn)

def expectedFailureCUDA(fn):
    return expectedFailure('cuda')(fn)

def expectedFailureMeta(fn):
    return expectedFailure('meta')(fn)

class expectedAlertNondeterministic:
    def __init__(self, caller_name, device_type=None, fn_has_device_arg=True):
        self.device_type = device_type
        self.error_message = caller_name + ' does not have a deterministic implementation, but you set'
        self.fn_has_device_arg = fn_has_device_arg

    def __call__(self, fn):
        @wraps(fn)
        def efail_fn(slf, device, *args, **kwargs):
            with DeterministicGuard(True):

                if self.device_type is None or self.device_type == slf.device_type:
                    try:
                        if self.fn_has_device_arg:
                            fn(slf, device, *args, **kwargs)
                        else:
                            fn(slf, *args, **kwargs)
                    except RuntimeError as e:
                        if self.error_message not in str(e):
                            slf.fail(
                                'expected non-deterministic error message to start with "'
                                + self.error_message
                                + '" but got this instead: "' + str(e) + '"')
                        return
                    else:
                        slf.fail('expected a non-deterministic error, but it was not raised')

                try:
                    if self.fn_has_device_arg:
                        return fn(slf, device, *args, **kwargs)
                    else:
                        return fn(slf, *args, **kwargs)
                except RuntimeError as e:
                    if 'does not have a deterministic implementation' in str(e):
                        slf.fail(
                            'did not expect non-deterministic error message, '
                            + 'but got this: "' + str(e) + '"')

                    raise

        @wraps(fn)
        def efail_fn_no_device(slf, *args, **kwargs):
            return efail_fn(slf, None, *args, **kwargs)

        if self.fn_has_device_arg:
            return efail_fn
        else:
            return efail_fn_no_device

def skipCPUIfNoLapack(fn):
    return skipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)

def skipCPUIfNoFFT(fn):
    return skipCPUIf(not torch._C.has_spectral, "PyTorch is built without FFT support")(fn)

def skipCPUIfNoMkl(fn):
    return skipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)

def skipCUDAIfNoMagma(fn):
    return skipCUDAIf('no_magma', "no MAGMA library detected")(skipCUDANonDefaultStreamIf(True)(fn))

def skipCUDAIfNoCusolver(fn):
    version = _get_torch_cuda_version()
    return skipCUDAIf(version < (10, 2), "cuSOLVER not available")(fn)

def skipCUDAIfNoMagmaAndNoCusolver(fn):
    version = _get_torch_cuda_version()
    if version >= (10, 2):
        return fn
    else:

        return skipCUDAIfNoMagma(fn)

def skipCUDAIfRocm(fn):
    return skipCUDAIf(TEST_WITH_ROCM, "test doesn't currently work on the ROCm stack")(fn)

def skipCUDAIfNotRocm(fn):
    return skipCUDAIf(not TEST_WITH_ROCM, "test doesn't currently work on the CUDA stack")(fn)

def skipCUDAIfRocmVersionLessThan(version=None):

    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if self.device_type == 'cuda':
                if not TEST_WITH_ROCM:
                    reason = "ROCm not available"
                    raise unittest.SkipTest(reason)
                rocm_version = str(torch.version.hip)
                rocm_version = rocm_version.split("-")[0]
                rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
                if rocm_version_tuple is None or version is None or rocm_version_tuple < tuple(version):
                    reason = "ROCm {0} is available but {1} required".format(rocm_version_tuple, version)
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

def skipCUDAIfNotMiopenSuggestNHWC(fn):
    return skipCUDAIf(not TEST_WITH_MIOPEN_SUGGEST_NHWC, "test doesn't currently work without MIOpen NHWC activation")(fn)

def skipCUDAVersionIn(versions : List[Tuple[int, int]] = None):
    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            version = _get_torch_cuda_version()
            if version == (0, 0):
                return fn(self, *args, **kwargs)
            if version in (versions or []):
                reason = "test skipped for CUDA version {0}".format(version)
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

def skipCUDAIfCudnnVersionLessThan(version=0):

    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if self.device_type == 'cuda':
                if self.no_cudnn:
                    reason = "cuDNN not available"
                    raise unittest.SkipTest(reason)
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = "cuDNN version {0} is available but {1} required".format(self.cudnn_version, version)
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

def skipCUDAIfNoCudnn(fn):
    return skipCUDAIfCudnnVersionLessThan(0)(fn)

def skipMeta(fn):
    return skipMetaIf(True, "test doesn't work with meta tensors")(fn)
