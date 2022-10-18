import unittest
from os.path import join, dirname

from arekit.common.data import const
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.views.linkages.multilabel import MultilableOpinionLinkagesView
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView
from arekit.contrib.utils.pipelines.opinion_collections import \
    text_opinion_linkages_to_opinion_collections_pipeline_part
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from tests.contrib.networks.test_tf_input_features import RuSentRelSynonymsCollectionProvider
from tests.contrib.networks.labels import TestThreeLabelScaler


class TestOutputFormatters(unittest.TestCase):

    __current_dir = dirname(__file__)
    __input_samples_filepath = join(__current_dir, "test_data/sample-train.tsv.gz")
    __input_opinions_filepath = join(__current_dir, "test_data/opinion-train.tsv.gz")
    __output_filepath = join(__current_dir, "test_data/output.tsv.gz")

    def test_output_formatter(self):

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
