import copy
import warnings
from torch.utils.data import IterDataPipe

class IterableWrapperIterDataPipe(IterDataPipe):
    r""":class:`IterableWrapperIterDataPipe`.

    Iterable datapipe that wraps an iterable object.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iteration.

    .. note::
      If `deepcopy` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.
    """
    def __init__(self, iterable, deepcopy=True):
        self.iterable = iterable
        self.deepcopy = deepcopy

    def __iter__(self):
        source_data = self.iterable
        if self.deepcopy:
            try:
                source_data = copy.deepcopy(self.iterable)

            except TypeError:
                warnings.warn(
                    "The input iterable can not be deepcopied, "
                    "please be aware of in-place modification would affect source data"
                )
        for data in source_data:
            yield data

    def __len__(self):
        return len(self.iterable)
