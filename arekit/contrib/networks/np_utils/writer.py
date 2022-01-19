import logging

import numpy as np

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.core.base_writer import BaseWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NpzDataWriter(BaseWriter):

    def write(self, data, target):
        assert(isinstance(target, str))
        logger.info("Save: {}".format(target))
        create_dir_if_not_exists(target)
        np.savez(target, data)
