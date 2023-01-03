import logging

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VocabRepositoryUtils(object):

    @staticmethod
    def save(data, target):
        logger.info("Saving vocabulary [size={size}]: {filepath}".format(size=len(data), filepath=target))
        np.savetxt(target, data, fmt='%s')

    @staticmethod
    def load(source):
        vocab = np.loadtxt(source, dtype=str, comments=None)
        logger.info("Loading vocabulary [size={size}]: {filepath}".format(size=len(vocab), filepath=source))
        return vocab
