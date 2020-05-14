from collections import OrderedDict
from os import path

import numpy as np
import pandas as pd

import io_utils
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter
from arekit.contrib.bert.formatters.utils import get_output_dir, generate_filename


# TODO. Inherit from base.
class BertOpinionsFormatter(object):

    ROW_ID = 'row_id'
    ID = 'id'
    SOURCE = 'source'
    TARGET = 'target'

    # region methods

    def __init__(self, data_type):
        assert(isinstance(data_type, unicode))
        self.__data_type = data_type
        self.__df = BertOpinionsFormatter.__create_empty_df()

    @staticmethod
    def __create_empty_df():
        dtypes_list = []
        dtypes_list.append((BertOpinionsFormatter.ROW_ID, 'int32'))
        dtypes_list.append((BertOpinionsFormatter.ID, unicode))
        dtypes_list.append((BertOpinionsFormatter.SOURCE, unicode))
        dtypes_list.append((BertOpinionsFormatter.TARGET, unicode))

        data = np.empty(0, dtype=np.dtype(dtypes_list))
        return pd.DataFrame(data)

    @staticmethod
    def __create_opinion_row(opinion_provider, linked_wrapper):
        """
        row format: [id, src, target, label]
        """
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_wrapper, LinkedTextOpinionsWrapper))

        row = OrderedDict()

        src_value = TextOpinionHelper.extract_entity_value(
            text_opinion=linked_wrapper.First,
            end_type=EntityEndType.Source)

        target_value = TextOpinionHelper.extract_entity_value(
            text_opinion=linked_wrapper.First,
            end_type=EntityEndType.Target)

        row[BertOpinionsFormatter.ID] = MultipleIDFormatter.create_opinion_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_wrapper,
            index_in_linked=0)
        row[BertOpinionsFormatter.SOURCE] = src_value
        row[BertOpinionsFormatter.TARGET] = target_value

        return row

    # TODO. Make protected
    @staticmethod
    def __iter_by_rows(opinion_provider):
        assert(isinstance(opinion_provider, OpinionProvider))

        linked_iter = opinion_provider.iter_linked_opinion_wrappers(balance=False,
                                                                    supported_labels=None)

        for linked_wrapper in linked_iter:
            yield BertOpinionsFormatter.__create_opinion_row(
                opinion_provider=opinion_provider,
                linked_wrapper=linked_wrapper)

    # TODO. To Base
    def __set_value(self, row_ind, column, value):
        self.__df.at[row_ind, column] = value

    # TODO. To Base
    def __fill_with_blank_rows(self, rows_count):
        assert(isinstance(rows_count, int))
        self.__df[self.ROW_ID] = range(rows_count)
        self.__df.set_index(self.ROW_ID, inplace=True)

    # endregion

    def provide_opinion_info_by_opinion_id(self, opinion_id):
        assert(isinstance(opinion_id, unicode))

        opinion_row = self.__df[self.__df[self.ID] == opinion_id]
        df_row = opinion_row.iloc[0].tolist()

        news_id = df_row[0]
        source = df_row[1].decode('utf-8')
        target = df_row[2].decode('utf-8')

        return news_id, source, target

    # TODO. To Base
    # TODO. To Base
    # TODO. To Base
    def format(self, opinion_provider):
        assert(isinstance(opinion_provider, OpinionProvider))

        rows_count = sum(1 for _ in self.__iter_by_rows(opinion_provider))

        self.__fill_with_blank_rows(rows_count)
        for row_index, row in enumerate(self.__iter_by_rows(opinion_provider)):
            for column, value in row.iteritems():
                self.__set_value(row_ind=row_index,
                                 column=column,
                                 value=value)

            current_work = row_index + 1
            total_work = rows_count
            percent = round(100 * float(current_work) / total_work, 2)
            print "Opinions ('{}') added: {}/{} ({}%)".format(self.__data_type, current_work, total_work, percent)

    def to_tsv_by_experiment(self, experiment):
        assert(isinstance(experiment, BaseExperiment))

        filepath = BertOpinionsFormatter.get_filepath(data_type=self.__data_type,
                                                      experiment=experiment)

        self.__df.to_csv(filepath,
                         sep='\t',
                         encoding='utf-8',
                         index=False,
                         header=False)

    def from_tsv(self, experiment):
        assert(isinstance(experiment, BaseExperiment))

        filepath = BertOpinionsFormatter.get_filepath(data_type=self.__data_type,
                                                      experiment=experiment)

        self.__df = pd.read_csv(filepath,
                                sep='\t',
                                header=None,
                                names=[self.ID, self.SOURCE, self.TARGET])

    def get_ids(self):
        return self.__df.iloc[0].tolist()

    @staticmethod
    def get_filepath(data_type, experiment):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(data_type, unicode))

        fname = generate_filename(data_type=data_type,
                                  experiment=experiment,
                                  prefix=u'opinions')

        filepath = path.join(get_output_dir(experiment=experiment), fname)

        io_utils.create_dir_if_not_exists(filepath)

        return filepath

