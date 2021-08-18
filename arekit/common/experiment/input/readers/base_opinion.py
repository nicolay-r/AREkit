from arekit.common.experiment import const
from arekit.common.experiment.input.readers.base import BaseInputReader


class BaseInputOpinionReader(BaseInputReader):

    def provide_opinion_info_by_opinion_id(self, opinion_id):
        assert(isinstance(opinion_id, str))

        opinion_row = self._df[self._df[const.ID] == opinion_id]
        df_row = opinion_row.iloc[0]

        source = df_row[const.SOURCE]
        target = df_row[const.TARGET]

        return source, target
