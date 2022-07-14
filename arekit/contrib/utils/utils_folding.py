from arekit.common.folding.base import BaseDataFolding
from arekit.contrib.utils.cv.two_class import TwoClassCVFolding


def folding_iter_states(folding):
    if isinstance(folding, TwoClassCVFolding):
        for state in folding.iter_states():
            yield state
    yield 0


def experiment_iter_index(folding):
    assert(isinstance(folding, BaseDataFolding))

    if isinstance(folding, TwoClassCVFolding):
        return folding.StateIndex

    # In other cases we consider that there is only a single state.
    return 0