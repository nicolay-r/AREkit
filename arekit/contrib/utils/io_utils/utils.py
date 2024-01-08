from collections.abc import Iterable
import logging
from os.path import exists


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_targets_existence(targets):
    assert (isinstance(targets, Iterable))

    result = True
    for filepath in targets:
        assert(isinstance(filepath, str))

        existed = exists(filepath)
        logger.info("Check existence [{is_existed}]: {fp}".format(is_existed=existed, fp=filepath))
        if not existed:
            result = False

    return result
