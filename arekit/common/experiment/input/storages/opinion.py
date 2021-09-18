import logging
from arekit.common.experiment.input.storages.base import BaseRowsStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseOpinionsStorage(BaseRowsStorage):

    def __init__(self, columns_provider):
        super(BaseOpinionsStorage, self).__init__(columns_provider)

    def save(self):
        """ In Memory solution, there is no need to write it.
        """
        pass
