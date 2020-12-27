import sys
import unittest

sys.path.append('../')

from arekit.contrib.source.rusentrel.synonyms_helper import RuSentRelSynonymsCollectionHelper
from arekit.contrib.bert.core.input.providers.label.binary import BinaryLabelProvider
from arekit.common.experiment.data_type import DataType
from arekit.contrib.experiments.common import entity_to_group_func
from arekit.common.experiment.input.formatters.helper.balancing import SampleRowBalancerHelper
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.processing.lemmatization.mystem import MystemWrapper


class TestInputBalancing(unittest.TestCase):

    def test_balancing(self):

        stemmer = MystemWrapper()
        label_provider = BinaryLabelProvider(label_scaler=ThreeLabelScaler())
        synonyms = RuSentRelSynonymsCollectionHelper.load_collection(stemmer=stemmer,
                                                                     is_read_only=True)
        terms_mapper = OpinionContainingTextTermsMapper(
            entity_formatter=StringEntitiesSimpleFormatter(),
            entity_to_group_func=lambda entity: entity_to_group_func(entity=entity,
                                                                     synonyms=synonyms))

        formatter = BaseSampleFormatter(
            data_type=DataType.Train,
            label_provider=label_provider,
            text_provider=BaseSingleTextProvider(terms_mapper),
            balance=False)

        df = formatter._df

        df = df.append({"row_id": 1, "id": 1, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 2, "label": 1, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 5, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 6, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)

        balanced_df = SampleRowBalancerHelper.calculate_balanced_df(
            df=df,
            create_blank_df=lambda size: formatter._create_blank_df(size),
            label_provider=label_provider)

        print "Original:"
        print df.shape

        print "Balanced:"
        print balanced_df.shape


if __name__ == '__main__':
    unittest.main()
