import collections

from arekit.common.entities.base import Entity
from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.frame_variants.parse import FrameVariantsParser
from arekit.common.labels.base import NeutralLabel
from arekit.common.linked.text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.model.sample import InputSampleBase
from arekit.common.news import News
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.processing.text.token import Token

NewsTermsShow = False
NewsTermsStatisticShow = False


# region private methods

def __debug_show_terms(parsed_news):
    assert(isinstance(parsed_news, ParsedNews))
    for term in parsed_news.iter_terms():
        if isinstance(term, unicode):
            print "Word:    '{}'".format(term.encode('utf-8'))
        elif isinstance(term, Token):
            print "Token:   '{}' ('{}')".format(term.get_token_value().encode('utf-8'),
                                                term.get_original_value().encode('utf-8'))
        elif isinstance(term, Entity):
            print "Entity:  '{}'".format(term.Value.encode('utf-8'))
        else:
            raise Exception("unsuported type {}".format(term))


def __debug_statistics(parsed_news):
    assert(isinstance(parsed_news, ParsedNews))

    terms = list(parsed_news.iter_terms())
    words = filter(lambda term: isinstance(term, unicode), terms)
    tokens = filter(lambda term: isinstance(term, Token), terms)
    entities = filter(lambda term: isinstance(term, Entity), terms)
    total = len(words) + len(tokens) + len(entities)

    print "Extracted news_words info, NEWS_ID: {}".format(parsed_news.RelatedNewsID)
    print "\tWords: {} ({}%)".format(len(words), 100.0 * len(words) / total)
    print "\tTokens: {} ({}%)".format(len(tokens), 100.0 * len(tokens) / total)
    print "\tEntities: {} ({}%)".format(len(entities), 100.0 * len(entities) / total)


def __iter_opinion_collections(opin_operations, doc_id, data_type):
    assert(isinstance(opin_operations, OpinionOperations))
    assert(isinstance(doc_id, int))
    assert(isinstance(data_type, unicode))

    neutral = opin_operations.read_neutral_opinion_collection(doc_id=doc_id,
                                                              data_type=data_type)

    if neutral is not None:
        yield neutral

    if data_type == DataType.Train:
        yield opin_operations.read_etalon_opinion_collection(doc_id=doc_id)


def __check_text_opinion(text_opinion, terms_per_context):
    assert(isinstance(text_opinion, TextOpinion))
    return InputSampleBase.check_ability_to_create_sample(
        window_size=terms_per_context,
        text_opinion=text_opinion)


def __create_parsed_collection(doc_operations, data_io, data_type):
    assert(isinstance(doc_operations, DocumentOperations))
    assert(isinstance(data_io, DataIO))
    assert(isinstance(data_type, unicode))

    parsed_collection = ParsedNewsCollection()

    for doc_id in doc_operations.iter_news_indices(data_type):

        news = doc_operations.read_news(doc_id=doc_id)
        parsed_news = news.parse(options=doc_operations.create_parse_options())

        assert(isinstance(news, News))

        if NewsTermsStatisticShow:
            __debug_statistics(parsed_news)
        if NewsTermsShow:
            __debug_show_terms(parsed_news)

        parsed_news.modify_parsed_sentences(
            lambda sentence: FrameVariantsParser.parse_frames_in_parsed_text(
                frame_variants_collection=data_io.FrameVariantCollection,
                parsed_text=sentence))

        if not parsed_collection.contains_id(doc_id):
            parsed_collection.add(parsed_news)
        else:
            print "Warning: Skipping document with id={}, news={}".format(news.ID, news)

    return parsed_collection

# endregions


def extract_text_opinions_and_parse_news(experiment,
                                         data_type,
                                         terms_per_context):
    """
    Extracting text-level opinions based on doc-level opinions in documents,
    obtained by information in experiment.

    NOTE:
    1. Assumes to provide the same label (doc level opinion) onto related text-level opinions.
    """
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, unicode))
    assert(isinstance(terms_per_context, int))
    assert(terms_per_context > 0)

    parsed_collection = __create_parsed_collection(
        doc_operations=experiment.DocumentOperations,
        data_io=experiment.DataIO,
        data_type=data_type)

    text_opinions = LabeledLinkedTextOpinionCollection(
        parsed_news_collection=parsed_collection)

    for doc_id in parsed_collection.iter_news_ids():

        news = experiment.DocumentOperations.read_news(doc_id=doc_id)
        assert(isinstance(news, News))

        opinions_it = __iter_opinion_collections(opin_operations=experiment.OpinionOperations,
                                                 doc_id=doc_id,
                                                 data_type=data_type)

        for opinions in opinions_it:
            for linked_wrap in news.iter_wrapped_linked_text_opinions(opinions=opinions):
                text_opinions.try_add_linked_text_opinions(
                    linked_text_opinions=linked_wrap,
                    check_opinion_correctness=lambda text_opinion: __check_text_opinion(text_opinion, terms_per_context))

    return text_opinions


def compose_opinion_collection(create_collection_func,
                               wrapped_linked_opinion_iter,
                               labels_helper,
                               label_calc_mode):
    assert(callable(create_collection_func))
    assert(isinstance(wrapped_linked_opinion_iter, collections.Iterable))
    assert(isinstance(labels_helper, LabelsHelper))

    collection = create_collection_func()
    assert(isinstance(collection, OpinionCollection))

    for linked_opins in wrapped_linked_opinion_iter:

        label = labels_helper.aggregate_labels(
            labels_list=[opinion.Sentiment for opinion in linked_opins],
            label_creation_mode=label_calc_mode)

        agg_opinion = linked_opins.aggregate_data(label=label)

        assert(isinstance(agg_opinion, Opinion))

        if isinstance(agg_opinion.Sentiment, NeutralLabel):
            continue

        if collection.has_synonymous_opinion(agg_opinion):
            continue

        collection.add_opinion(agg_opinion)

    return collection
