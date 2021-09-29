from arekit.common.experiment import const
from arekit.common.experiment.input.readers.base import BaseInputReader


class BaseInputOpinionReader(BaseInputReader):

    def provide_opinion_info_by_opinion_id(self, opinion_id):
        assert(isinstance(opinion_id, str))

        row = self._storage.find_first_by_value(column_name=const.ID,
                                                value=opinion_id)

        source = row[const.SOURCE]
        target = row[const.TARGET]

        return source, target
