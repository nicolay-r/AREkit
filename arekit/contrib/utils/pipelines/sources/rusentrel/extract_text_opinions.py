from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.algo.predefined import PredefinedOpinionAnnotationAlgorithm
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinions
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollectionHelper
from arekit.contrib.utils.pipelines.sources.rusentrel.doc_provider import RuSentrelDocumentProvider
from arekit.contrib.utils.pipelines.text_opinion.annot.algo_based import AlgorithmBasedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


def create_text_opinion_extraction_pipeline(rusentrel_version,
                                            text_parser,
                                            labels_fmt,
                                            custom_text_opinion_filters=None,
                                            no_label=NoLabel(),
                                            terms_per_context=50,
                                            dist_in_sentences=0):
    """ Processing pipeline for RuSentRel, which combines:
            - predefined document-level annotation (sentiment labels)
            - automatic annotation of opinions between mentioned named entities (no-label)

        Original collection paper: arxiv.org/abs/1808.08932

        version: enum
            Version of the RuSentRel collection.
        terms_per_context: int
            Amount of terms that we consider in between the Object and Subject.
        dist_in_sentences: int
            considering amount of sentences that could be in between Object and Subject.
    """
    assert(isinstance(labels_fmt, RuSentRelLabelsFormatter))
    assert(isinstance(custom_text_opinion_filters, list) or custom_text_opinion_filters is None)

    synonyms = StemmerBasedSynonymCollection(
        iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(rusentrel_version),
        stemmer=MystemWrapper(),
        is_read_only=False)

    doc_provider = RuSentrelDocumentProvider(version=rusentrel_version, synonyms=synonyms)

    text_opinion_filters = [
        DistanceLimitedTextOpinionFilter(terms_per_context)
    ]

    # Append with the custom filters afterwards.
    if custom_text_opinion_filters is not None:
        text_opinion_filters += custom_text_opinion_filters

    pipeline = text_opinion_extraction_pipeline(
        annotators=[
            predefined_annotator(synonyms=synonyms, labels_fmt=labels_fmt),
            nolabel_annotator(synonyms=synonyms, terms_per_context=terms_per_context,
                              dist_in_sentences=dist_in_sentences, no_label=no_label)
        ],
        text_opinion_filters=text_opinion_filters,
        get_doc_by_id_func=doc_provider.by_id,
        entity_index_func=lambda brat_entity: brat_entity.ID,
        text_parser=text_parser)

    return pipeline


def nolabel_annotator(synonyms, terms_per_context, dist_in_sentences=0, no_label=NoLabel()):
    """ This is a default annotator, utilized to annotate `neutral`-like attitudes.
        Neutral means that we adopt no-label parameter, and this label might be customized
        to the one required in your studies.
    """
    return AlgorithmBasedTextOpinionAnnotator(
        annot_algo=PairBasedOpinionAnnotationAlgorithm(dist_in_sents=dist_in_sentences,
                                                       dist_in_terms_bound=terms_per_context,
                                                       label_provider=ConstantLabelProvider(no_label),
                                                       entity_index_func=lambda brat_entity: brat_entity.ID),
        create_empty_collection_func=lambda: OpinionCollection(
            synonyms=synonyms, error_on_duplicates=True, error_on_synonym_end_missed=False),
        value_to_group_id_func=lambda value:
        SynonymsCollectionValuesGroupingProviders.provide_existed_value(synonyms=synonyms, value=value))


def predefined_annotator(synonyms, labels_fmt):
    """ This is a annotator-converter of the predefined Document-Level opinions onto text-level one
    """
    return AlgorithmBasedTextOpinionAnnotator(
        annot_algo=PredefinedOpinionAnnotationAlgorithm(
            lambda doc_id: __get_document_opinions(doc_id=doc_id, synonyms=synonyms, labels_fmt=labels_fmt)),
        create_empty_collection_func=lambda: OpinionCollection(
            synonyms=synonyms, error_on_duplicates=True, error_on_synonym_end_missed=False),
        value_to_group_id_func=lambda value:
        SynonymsCollectionValuesGroupingProviders.provide_existed_value(synonyms=synonyms, value=value))


def __get_document_opinions(doc_id, synonyms, labels_fmt):
    """ RuSentRel provides a pre-defined list of Document-Level Opinions.
        Within this function we create the related OpinionCollection by a given doc_id.
    """
    return OpinionCollection(
        opinions=RuSentRelOpinions.iter_from_doc(
            doc_id=doc_id, labels_fmt=labels_fmt),
        synonyms=synonyms,
        error_on_synonym_end_missed=True,
        error_on_duplicates=True)
