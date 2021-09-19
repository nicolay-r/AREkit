import logging
from arekit.common.experiment.input.storages.base import BaseRowsStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseOpinionsStorage(BaseRowsStorage):

    def __init__(self):
        super(BaseOpinionsStorage, self).__init__()

    def save(self, target):
        """ In Memory solution, there is no need to write it.
        """
        super(BaseOpinionsStorage, self).save(target)
