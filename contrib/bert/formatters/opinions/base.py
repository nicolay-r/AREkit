from collections import OrderedDict
from os import path

import numpy as np
import pandas as pd

import io_utils
from arekit.common.linked.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper

from arekit.common.experiment.base import BaseExperiment
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter
from arekit.contrib.bert.formatters.utils import get_output_dir, generate_filename


class BertOpinionsFormatter(object):

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
        dtypes_list.append((BertOpinionsFormatter.ID, 'int32'))
        dtypes_list.append((BertOpinionsFormatter.SOURCE, 'string'))
        dtypes_list.append((BertOpinionsFormatter.TARGET, 'string'))

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
            text_opinion=linked_wrapper.FirstOpinion,
            end_type=EntityEndType.Source)

        target_value = TextOpinionHelper.extract_entity_value(
            text_opinion=linked_wrapper.FirstOpinion,
            end_type=EntityEndType.Target)

        row[BertOpinionsFormatter.ID] = MultipleIDFormatter.create_opinion_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_wrapper,
            index_in_linked=0)
        row[BertOpinionsFormatter.SOURCE] = src_value
        row[BertOpinionsFormatter.TARGET] = target_value

        return row

    # endregion

    def provide_opinion_info_by_opinion_id(self, opinion_id):
        assert(isinstance(opinion_id, unicode))

        opinion_row = self.__df[self.__df[self.ID] == opinion_id]
        df_row = opinion_row.iloc[0].tolist()

        news_id = df_row[0]
        source = df_row[1].decode('utf-8')
        target = df_row[2].decode('utf-8')

        return news_id, source, target

    def format(self, opinion_provider):
        assert(isinstance(opinion_provider, OpinionProvider))

        print "Adding opinions ('{}') ... ".format(self.__data_type)

        linked_iter = opinion_provider.iter_linked_opinion_wrappers(
            balance=False,
            supported_labels=None)

        for linked_wrapper in linked_iter:
            row = BertOpinionsFormatter.__create_opinion_row(
                opinion_provider=opinion_provider,
                linked_wrapper=linked_wrapper)
            self.__df = self.__df.append(row, ignore_index=True)

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

