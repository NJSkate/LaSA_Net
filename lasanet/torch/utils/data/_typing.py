
import collections
import numbers
import sys
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
                    get_type_hints)
from typing import _eval_type, _tp_cache, _type_check, _type_repr

try:
    from typing import ForwardRef
except ImportError:
    from typing import _ForwardRef as ForwardRef

try:
    from typing import GenericMeta
    _GenericAlias = GenericMeta
except ImportError:

    from abc import ABCMeta
    from typing import _ProtocolMeta, _GenericAlias

    class GenericMeta(_ProtocolMeta, ABCMeta):
        pass

class Integer(numbers.Integral):
    pass

class Boolean(numbers.Integral):
    pass

TYPE2ABC = {
    bool: Boolean,
    int: Integer,
    float: numbers.Real,
    complex: numbers.Complex,
    dict: Dict,
    list: List,
    set: Set,
    tuple: Tuple,
    None: type(None),
}

def issubtype(left, right, recursive=True):
    r"""
    Check if the left-side type is a subtype of the right-side type.
    If any of type is a composite type like `Union` and `TypeVar` with
    bounds, it would be expanded into a list of types and check all
    of left-side types are subtypes of either one from right-side types.
    """
    left = TYPE2ABC.get(left, left)
    right = TYPE2ABC.get(right, right)

    if right is Any or left == right:
        return True

    if isinstance(right, _GenericAlias):
        if getattr(right, '__origin__', None) is Generic:
            return True

    if right == type(None):
        return False

    constraints = _decompose_type(right)

    if len(constraints) == 0 or Any in constraints:
        return True

    if left is Any:
        return False

    variants = _decompose_type(left)

    if len(variants) == 0:
        return False

    return all(_issubtype_with_constraints(variant, constraints, recursive) for variant in variants)

def _decompose_type(t, to_list=True):
    if isinstance(t, TypeVar):
        if t.__bound__ is not None:
            ts = [t.__bound__]
        else:

            ts = list(t.__constraints__)
    elif hasattr(t, '__origin__') and t.__origin__ == Union:
        ts = t.__args__
    else:
        if not to_list:
            return None
        ts = [t]

    ts = list(TYPE2ABC.get(_t, _t) for _t in ts)
    return ts

def _issubtype_with_constraints(variant, constraints, recursive=True):
    r"""
    Check if the variant is a subtype of either one from constraints.
    For composite types like `Union` and `TypeVar` with bounds, they
    would be expanded for testing.
    """
    if variant in constraints:
        return True

    vs = _decompose_type(variant, to_list=False)

    if vs is not None:
        return all(_issubtype_with_constraints(v, constraints, recursive) for v in vs)

    if hasattr(variant, '__origin__') and variant.__origin__ is not None:
        v_origin = variant.__origin__

        v_args = getattr(variant, "__args__", None)
    else:
        v_origin = variant
        v_args = None

    for constraint in constraints:
        cs = _decompose_type(constraint, to_list=False)

        if cs is not None:
            if _issubtype_with_constraints(variant, cs, recursive):
                return True

        else:

            if hasattr(constraint, '__origin__') and constraint.__origin__ is not None:
                c_origin = constraint.__origin__
                if v_origin == c_origin:
                    if not recursive:
                        return True

                    c_args = getattr(constraint, "__args__", None)
                    if c_args is None or len(c_args) == 0:
                        return True
                    if v_args is not None and len(v_args) == len(c_args) and\
                            all(issubtype(v_arg, c_arg) for v_arg, c_arg in zip(v_args, c_args)):
                        return True

            else:
                if v_origin == constraint:
                    return True

    return False

def issubinstance(data, data_type):
    if not issubtype(type(data), data_type, recursive=False):
        return False

    dt_args = getattr(data_type, "__args__", None)
    if isinstance(data, tuple):
        if dt_args is None or len(dt_args) == 0:
            return True
        if len(dt_args) != len(data):
            return False
        return all(issubinstance(d, t) for d, t in zip(data, dt_args))
    elif isinstance(data, (list, set)):
        if dt_args is None or len(dt_args) == 0:
            return True
        t = dt_args[0]
        return all(issubinstance(d, t) for d in data)
    elif isinstance(data, dict):
        if dt_args is None or len(dt_args) == 0:
            return True
        kt, vt = dt_args
        return all(issubinstance(k, kt) and issubinstance(v, vt) for k, v in data.items())

    return True

class _DataPipeType:
    r"""
    Save type annotation in `param`
    """

    def __init__(self, param):
        self.param = param

    def __repr__(self):
        return _type_repr(self.param)

    def __eq__(self, other):
        if isinstance(other, _DataPipeType):
            return self.param == other.param
        return NotImplemented

    def __hash__(self):
        return hash(self.param)

    def issubtype(self, other):
        if isinstance(other.param, _GenericAlias):
            if getattr(other.param, '__origin__', None) is Generic:
                return True
        if isinstance(other, _DataPipeType):
            return issubtype(self.param, other.param)
        if isinstance(other, type):
            return issubtype(self.param, other)
        raise TypeError("Expected '_DataPipeType' or 'type', but found {}".format(type(other)))

    def issubtype_of_instance(self, other):
        return issubinstance(other, self.param)

T_co = TypeVar('T_co', covariant=True)
_DEFAULT_TYPE = _DataPipeType(Generic[T_co])

class _DataPipeMeta(GenericMeta):
    r"""
    Metaclass for `DataPipe`. Add `type` attribute and `__init_subclass__` based
    on the type, and validate the return hint of `__iter__`.
    """
    type: _DataPipeType

    def __new__(cls, name, bases, namespace, **kwargs):

        cls.__origin__ = None

        if '_is_protocol' not in namespace:
            namespace['_is_protocol'] = True
        if 'type' in namespace:
            return super().__new__(cls, name, bases, namespace, **kwargs)

        namespace['__type_class__'] = False

        for base in bases:
            if isinstance(base, _DataPipeMeta):
                return super().__new__(cls, name, bases, namespace, **kwargs)

        namespace.update({'type': _DEFAULT_TYPE,
                          '__init_subclass__': _dp_init_subclass})
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __init__(self, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

    @_tp_cache
    def __getitem__(self, params):
        if params is None:
            raise TypeError('{}[t]: t can not be None'.format(self.__name__))
        if isinstance(params, str):
            params = ForwardRef(params)
        if not isinstance(params, tuple):
            params = (params, )

        msg = "{}[t]: t must be a type".format(self.__name__)
        params = tuple(_type_check(p, msg) for p in params)

        if isinstance(self.type.param, _GenericAlias):
            orig = getattr(self.type.param, '__origin__', None)
            if isinstance(orig, type) and orig is not Generic:
                p = self.type.param[params]
                t = _DataPipeType(p)
                l = len(str(self.type)) + 2
                name = self.__name__[:-l]
                name = name + '[' + str(t) + ']'
                bases = (self,) + self.__bases__
                return self.__class__(name, bases,
                                      {'__init_subclass__': _dp_init_subclass,
                                       'type': t,
                                       '__type_class__': True})

        if len(params) > 1:
            raise TypeError('Too many parameters for {} actual {}, expected 1'.format(self, len(params)))

        t = _DataPipeType(params[0])

        if not t.issubtype(self.type):
            raise TypeError('Can not subclass a DataPipe[{}] from DataPipe[{}]'
                            .format(t, self.type))

        if self.type == t:
            return self

        name = self.__name__ + '[' + str(t) + ']'
        bases = (self,) + self.__bases__

        return self.__class__(name, bases,
                              {'__init_subclass__': _dp_init_subclass,
                               '__type_class__': True,
                               'type': t})

    def __eq__(self, other):
        if not isinstance(other, _DataPipeMeta):
            return NotImplemented
        if self.__origin__ is None or other.__origin__ is None:
            return self is other
        return (self.__origin__ == other.__origin__
                and self.type == other.type)

    def __hash__(self):
        return hash((self.__name__, self.type))

def _dp_init_subclass(sub_cls, *args, **kwargs):

    sub_cls.reinforce_type = reinforce_type

    if getattr(sub_cls, '__type_class__', False):
        return

    if isinstance(sub_cls.type.param, ForwardRef):
        base_globals = sys.modules[sub_cls.__module__].__dict__
        try:
            param = _eval_type(sub_cls.type.param, base_globals, locals())
            sub_cls.type.param = param
        except TypeError as e:
            raise TypeError("{} is not supported by Python typing"
                            .format(sub_cls.type.param.__forward_arg__)) from e

    if '__iter__' in sub_cls.__dict__:
        iter_fn = sub_cls.__dict__['__iter__']
        hints = get_type_hints(iter_fn)
        if 'return' in hints:
            return_hint = hints['return']

            if return_hint == Iterator:
                return
            if not (hasattr(return_hint, '__origin__') and
                    (return_hint.__origin__ == Iterator or
                     return_hint.__origin__ == collections.abc.Iterator)):
                raise TypeError("Expected 'Iterator' as the return annotation for `__iter__` of {}"
                                ", but found {}".format(sub_cls.__name__, _type_repr(hints['return'])))
            data_type = return_hint.__args__[0]
            if not issubtype(data_type, sub_cls.type.param):
                raise TypeError("Expected return type of '__iter__' as a subtype of {}, but found {}"
                                " for {}".format(sub_cls.type, _type_repr(data_type), sub_cls.__name__))

def reinforce_type(self, expected_type):
    r"""
    Reinforce the type for DataPipe instance. And the 'expected_type' is required
    to be a subtype of the original type hint to restrict the type requirement
    of DataPipe instance.
    """
    if isinstance(expected_type, tuple):
        expected_type = Tuple[expected_type]
    _type_check(expected_type, msg="'expected_type' must be a type")

    if not issubtype(expected_type, self.type.param):
        raise TypeError("Expected 'expected_type' as subtype of {}, but found {}"
                        .format(self.type, _type_repr(expected_type)))

    self.type = _DataPipeType(expected_type)
    return self
