import os
import unittest
from os.path import dirname, join

from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter

from tests.tutorials.test_tutorial_pipeline_sampling_bert import CustomEntitiesFormatter, SentimentLabelScaler, \
    CustomLabelsFormatter, Positive, Negative
from tests.tutorials.test_tutorial_pipeline_text_opinion_annotation import FooDocumentOperations


class TestStreamWriters(unittest.TestCase):

    __output_dir = join(dirname(__file__), "out")

    def __launch(self, writer, target_extention):
        assert(isinstance(target_extention, str))

        text_b_template = '{subject} к {object} в контексте : << {context} >>'

        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir)

        terms_mapper = BertDefaultStringTextTermsMapper(
            entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O"))

        text_provider = BaseSingleTextProvider(terms_mapper) \
            if text_b_template is None else \
            PairTextProvider(text_b_template, terms_mapper)

        sample_rows_provider = BaseSampleRowProvider(
            label_provider=MultipleLabelProvider(SentimentLabelScaler()),
            text_provider=text_provider)

        samples_io = SamplesIO(self.__output_dir, writer, target_extension=target_extention)

        pipeline_item = BertExperimentInputSerializerPipelineItem(
            rows_provider=sample_rows_provider,
            samples_io=samples_io,
            save_labels_func=lambda data_type: True,
            balance_func=lambda _: False,
            storage=RowCacheStorage())

        pipeline = BasePipeline([
            pipeline_item
        ])

        #####
        # Declaring pipeline related context parameters.
        #####
        no_folding = NoFolding(doc_ids=[0, 1], supported_data_type=DataType.Train)
        doc_ops = FooDocumentOperations()
        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(), DefaultTextTokenizer(keep_tokens=True)])
        train_pipeline = text_opinion_extraction_pipeline(
            annotators=[
                PredefinedTextOpinionAnnotator(
                    doc_ops,
                    label_formatter=CustomLabelsFormatter(pos_label_type=Positive,
                                                          neg_label_type=Negative))
            ],
            text_opinion_filters=[
                DistanceLimitedTextOpinionFilter(terms_per_context=50)
            ],
            get_doc_by_id_func=doc_ops.get_doc,
            text_parser=text_parser)
        #####

        pipeline.run(input_data=None,
                     params_dict={
                         "data_folding": no_folding,
                         "data_type_pipelines": {DataType.Train: train_pipeline}
                     })

    def test_csv_native(self):
        """ Testing writing into CSV format
        """
        return self.__launch(writer=NativeCsvWriter(), target_extention=".csv")

    def test_json_native(self):
        """ Testing writing into CSV format
        """
        return self.__launch(writer=OpenNREJsonWriter(text_columns=[BaseSingleTextProvider.TEXT_A]),
                             target_extention=".jsonl")
