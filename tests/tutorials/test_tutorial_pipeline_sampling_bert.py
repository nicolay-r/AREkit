import unittest
from collections import OrderedDict
from os.path import join, dirname

from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel, Label
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.storages.pandas_based import PandasBasedRowsStorage
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from tests.tutorials.test_tutorial_pipeline_text_opinion_annotation import FooDocumentProvider


class Positive(Label):
    pass


class Negative(Label):
    pass


class SentimentLabelScaler(BaseLabelScaler):

    def __init__(self):
        int_to_label = OrderedDict([(NoLabel(), 0), (Positive(), 1), (Negative(), -1)])
        uint_to_label = OrderedDict([(NoLabel(), 0), (Positive(), 1), (Negative(), 2)])
        super(SentimentLabelScaler, self).__init__(int_dict=int_to_label,
                                                   uint_dict=uint_to_label)


class CustomLabelsFormatter(StringLabelsFormatter):
    def __init__(self, pos_label_type, neg_label_type):
        stol = {"POSITIVE_TO": neg_label_type, "NEGATIVE_TO": pos_label_type}
        super(CustomLabelsFormatter, self).__init__(stol=stol)


class CustomEntitiesFormatter(StringEntitiesFormatter):

    def __init__(self, subject_fmt="[subject]", object_fmt="[object]"):
        self.__subj_fmt = subject_fmt
        self.__obj_fmt = object_fmt

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        if entity_type == OpinionEntityType.Other:
            return original_value.Value
        elif entity_type == OpinionEntityType.Object or entity_type == OpinionEntityType.SynonymObject:
            return self.__obj_fmt
        elif entity_type == OpinionEntityType.Subject or entity_type == OpinionEntityType.SynonymSubject:
            return self.__subj_fmt
        return None


class TestBertSerialization(unittest.TestCase):

    __output_dir = join(dirname(__file__), "out")

    def test(self):
        text_b_template = '{subject} к {object} в контексте : << {context} >>'

        terms_mapper = BertDefaultStringTextTermsMapper(
            entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O"))

        text_provider = BaseSingleTextProvider(terms_mapper) \
            if text_b_template is None else \
            PairTextProvider(text_b_template, terms_mapper)

        rows_provider = BaseSampleRowProvider(
            label_provider=MultipleLabelProvider(SentimentLabelScaler()),
            text_provider=text_provider)

        writer = PandasCsvWriter(write_header=True)
        samples_io = SamplesIO(self.__output_dir, writer, target_extension=".tsv.gz")

        pipeline_item = BertExperimentInputSerializerPipelineItem(
            rows_provider=rows_provider,
            samples_io=samples_io,
            save_labels_func=lambda data_type: True,
            storage=PandasBasedRowsStorage())

        pipeline = BasePipeline([
            pipeline_item
        ])

        #####
        # Declaring pipeline related context parameters.
        #####
        doc_provider = FooDocumentProvider()
        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(), DefaultTextTokenizer(keep_tokens=True)])
        train_pipeline = text_opinion_extraction_pipeline(
            annotators=[
                PredefinedTextOpinionAnnotator(
                    doc_provider,
                    label_formatter=CustomLabelsFormatter(pos_label_type=Positive,
                                                          neg_label_type=Negative))
            ],
            text_opinion_filters=[
                DistanceLimitedTextOpinionFilter(terms_per_context=50)
            ],
            get_doc_by_id_func=doc_provider.by_id,
            text_parser=text_parser)
        #####

        pipeline.run(input_data=None,
                     params_dict={
                         "data_folding": NoFolding(),
                         "data_type_pipelines": {DataType.Train: train_pipeline},
                         "doc_ids": {DataType.Train: [0, 1]}
                     })

        reader = PandasCsvReader()
        source = join(self.__output_dir, "sample-train-0.tsv.gz")
        storage = reader.read(source)
        self.assertEqual(28, len(storage), "Amount of rows is non equal!")
