import unittest
from os.path import join, dirname

from arekit.common.data import const
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.stemmer import Stemmer
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollectionHelper
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.views.linkages.multilabel import MultilableOpinionLinkagesView
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView
from arekit.contrib.utils.pipelines.opinion_collections import \
    text_opinion_linkages_to_opinion_collections_pipeline_part
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from collections import OrderedDict

from arekit.common.labels.base import NoLabel, Label
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


class TestNeutralLabel(NoLabel):
    pass


class TestPositiveLabel(Label):
    pass


class TestNegativeLabel(Label):
    pass


class TestThreeLabelScaler(SentimentLabelScaler):

    def __init__(self):

        uint_labels = [(TestNeutralLabel(), 0),
                       (TestPositiveLabel(), 1),
                       (TestNegativeLabel(), 2)]

        int_labels = [(TestNeutralLabel(), 0),
                      (TestPositiveLabel(), 1),
                      (TestNegativeLabel(), -1)]

        super(TestThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                                   int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)


class RuSentRelSynonymsCollectionProvider(object):

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuSentRelVersions.V11):
        assert(isinstance(stemmer, Stemmer))
        return StemmerBasedSynonymCollection(
            iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(version),
            debug=debug,
            stemmer=stemmer,
            is_read_only=is_read_only)


class TestOutputFormatters(unittest.TestCase):

    __current_dir = dirname(__file__)
    __input_opinions_filepath = join(__current_dir, "test_data/opinion-train.tsv.gz")
    __output_filepath = join(__current_dir, "test_data/output.tsv.gz")

    def optional_test_output_formatter(self):

        stemmer = MystemWrapper()
        synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer)

        label_scaler = TestThreeLabelScaler()

        reader = PandasCsvReader()

        output_storage = reader.read(target=self.__output_filepath)
        linkages_view = MultilableOpinionLinkagesView(labels_scaler=label_scaler, storage=output_storage)

        opinion_storage = reader.read(target=self.__input_opinions_filepath)
        opinion_view = BaseOpinionStorageView(opinion_storage)

        converter_part = text_opinion_linkages_to_opinion_collections_pipeline_part(
            create_opinion_collection_func=lambda: OpinionCollection(opinions=[],
                                                                     synonyms=synonyms,
                                                                     error_on_duplicates=True,
                                                                     error_on_synonym_end_missed=True),
            doc_ids_set={1},
            labels_scaler=label_scaler,
            iter_opinion_linkages_func=lambda doc_id: linkages_view.iter_opinion_linkages(
                doc_id=doc_id,
                opinions_view=opinion_view),
            label_calc_mode=LabelCalculationMode.AVERAGE)

        pipeline = BasePipeline(converter_part)

        doc_ids = set(opinion_storage.iter_column_values(column_name=const.DOC_ID, dtype=int))

        # Iterate over the result.
        for doc_id, collection in pipeline.run(doc_ids):
            print("d:{}, ct:{}, count:{}".format(doc_id, type(collection), len(collection)))


if __name__ == '__main__':
    unittest.main()
