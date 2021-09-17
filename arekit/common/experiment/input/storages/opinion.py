import logging
from arekit.common.experiment import const
from arekit.common.experiment.input.storages.base import BaseRowsStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseOpinionsStorage(BaseRowsStorage):

    def __init__(self):
        super(BaseOpinionsStorage, self).__init__()

    def _get_columns_list_with_types(self):
        dtypes_list = super(BaseOpinionsStorage, self)._get_columns_list_with_types()
        dtypes_list.append((const.ID, str))
        dtypes_list.append((const.NEWS_ID, 'int32'))
        dtypes_list.append((const.SOURCE, str))
        dtypes_list.append((const.TARGET, str))
        return dtypes_list

    def save(self):
        """ In Memory solution, there is no need to write it.
        """
        pass
