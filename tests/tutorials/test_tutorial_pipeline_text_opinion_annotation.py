import unittest

from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.labels.base import Label, NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider, EntityEndType
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.text_opinion.annot.algo_based import AlgorithmBasedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from tests.tutorials.test_tutorial_collection_binding import FooDocReader


class PositiveLabel(Label):
    pass


class NegativeLabel(Label):
    pass


class FooDocumentOperations(DocumentOperations):
    def by_id(self, doc_id):
        return FooDocReader.read_document(str(doc_id), doc_id=doc_id)


class CustomLabelsFormatter(StringLabelsFormatter):
    def __init__(self, pos_label_type, neg_label_type):
        stol = {"neg": neg_label_type, "pos": pos_label_type}
        super(CustomLabelsFormatter, self).__init__(stol=stol)


class TestTextOpinionAnnotation(unittest.TestCase):

    def test(self):
        doc_ops = FooDocumentOperations()
        predefined_annotator = PredefinedTextOpinionAnnotator(
            doc_ops=doc_ops,
            label_formatter=CustomLabelsFormatter(pos_label_type=PositiveLabel,
                                                  neg_label_type=NegativeLabel))

        synonyms = StemmerBasedSynonymCollection(stemmer=MystemWrapper(), is_read_only=False)

        nolabel_annotator = AlgorithmBasedTextOpinionAnnotator(
            annot_algo=PairBasedOpinionAnnotationAlgorithm(
                dist_in_sents=0,
                dist_in_terms_bound=50,
                label_provider=ConstantLabelProvider(NoLabel())),
            create_empty_collection_func=lambda: OpinionCollection(
                synonyms=synonyms, error_on_duplicates=True, error_on_synonym_end_missed=False),
            value_to_group_id_func=lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))

        text_parser = BaseTextParser([
            BratTextEntitiesParser(partitioning="string"),
            DefaultTextTokenizer(keep_tokens=True),
        ])

        pipeline = text_opinion_extraction_pipeline(
            annotators=[
                predefined_annotator,
                nolabel_annotator
            ],
            text_opinion_filters=[
                DistanceLimitedTextOpinionFilter(terms_per_context=50)
            ],
            get_doc_by_id_func=doc_ops.by_id,
            text_parser=text_parser)

        # Running the pipeline.
        for linked in pipeline.run(input_data=[0], params_dict={}):
            assert(isinstance(linked, TextOpinionsLinkage))

            pns = linked.Tag
            assert(isinstance(pns, ParsedNewsService))
            esp = pns.get_provider(EntityServiceProvider.NAME)
            source = esp.extract_entity_value(linked.First, EntityEndType.Source)
            target = esp.extract_entity_value(linked.First, EntityEndType.Target)

            print("`{}`->`{}`, {} [{}]".format(source, target,
                                               str(linked.First.Sentiment.__class__.__name__),
                                               len(linked)))


if __name__ == '__main__':
    unittest.main()
