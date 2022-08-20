from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollectionHelper
from arekit.contrib.utils.pipelines.source.rusentrel.doc_ops import RuSentrelDocumentOperations
from arekit.contrib.utils.pipelines.text_opinion.annot.algo_based import AlgorithmBasedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from arekit.contrib.utils.pipelines.text_opinion.filters.entity_based import EntityBasedTextOpinionFilter
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


def create_text_opinion_extraction_pipeline(rusentrel_version,
                                            text_parser,
                                            entity_filter=None,
                                            terms_per_context=50,
                                            dist_in_sentences=0):
    """ Processing pipeline for RuSentRel.

        Original collection paper: arxiv.org/abs/1808.08932

        version: enum
            Version of the RuSentRel collection.
        terms_per_context: int
            Amount of terms that we consider in between the Object and Subject.
        dist_in_sentences: int
            considering amount of sentences that could be in between Object and Subject.
    """

    synonyms = StemmerBasedSynonymCollection(
        iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(rusentrel_version),
        stemmer=MystemWrapper(),
        is_read_only=False,
        debug=False)

    doc_ops = RuSentrelDocumentOperations(version=rusentrel_version, synonyms=synonyms)

    algo_based_annotator = AlgorithmBasedTextOpinionAnnotator(
        annot_algo=PairBasedOpinionAnnotationAlgorithm(
            dist_in_sents=dist_in_sentences,
            dist_in_terms_bound=terms_per_context,
            label_provider=ConstantLabelProvider(NoLabel())),
        create_empty_collection_func=lambda: OpinionCollection(
            opinions=[],
            synonyms=synonyms,
            error_on_duplicates=True,
            error_on_synonym_end_missed=False),
        get_doc_existed_opinions_func=lambda _: None,
        value_to_group_id_func=lambda value:
        SynonymsCollectionValuesGroupingProviders.provide_existed_value(synonyms=synonyms, value=value))

    pipeline = text_opinion_extraction_pipeline(
        annotators=[
            PredefinedTextOpinionAnnotator(doc_ops),
            algo_based_annotator
        ],
        text_opinion_filters=[
            EntityBasedTextOpinionFilter(entity_filter=entity_filter),
            DistanceLimitedTextOpinionFilter(terms_per_context)
        ],
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser)

    return pipeline
