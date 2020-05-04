from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.model.sample import InputSampleBase
from arekit.common.news import News
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.frame_variants.parse import FrameVariantsParser
from arekit.common.model.helpers.parsed_news import ParsedNewsHelper


NewsTermsShow = False
NewsTermsStatisticShow = False

# region private methods


def __iter_opinion_collections(experiment, doc_id, data_type):
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(doc_id, int))
    assert(isinstance(data_type, unicode))

    neutral = experiment.read_neutral_opinion_collection(doc_id=doc_id,
                                                         data_type=data_type)

    if neutral is not None:
        yield neutral

    if data_type == DataType.Train:
        yield experiment.read_etalon_opinion_collection(doc_id=doc_id)


def __check_text_opinion(text_opinion, terms_per_context):
    assert(isinstance(text_opinion, TextOpinion))
    return InputSampleBase.check_ability_to_create_sample(
        window_size=terms_per_context,
        text_opinion=text_opinion)


def __create_parsed_collection(experiment, data_type):
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, unicode))

    parsed_collection = ParsedNewsCollection()

    for doc_id in experiment.iter_news_indices(data_type):

        news = experiment.read_news(doc_id=doc_id)
        parsed_news = news.parse(options=experiment.create_parse_options())

        assert(isinstance(news, News))

        if NewsTermsStatisticShow:
            ParsedNewsHelper.debug_statistics(parsed_news)
        if NewsTermsShow:
            ParsedNewsHelper.debug_show_terms(parsed_news)

        parsed_news.modify_parsed_sentences(
            lambda sentence: FrameVariantsParser.parse_frames_in_parsed_text(
                frame_variants_collection=experiment.DataIO.FrameVariantCollection,
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
        experiment=experiment,
        data_type=data_type)

    text_opinions = LabeledLinkedTextOpinionCollection(
        parsed_news_collection=parsed_collection)

    for doc_id in parsed_collection.iter_news_ids():

        news = experiment.read_news(doc_id=doc_id)

        opinions_it = __iter_opinion_collections(experiment=experiment,
                                                 doc_id=doc_id,
                                                 data_type=data_type)

        for opinions in opinions_it:
            for linked_wrap in news.iter_wrapped_linked_text_opinions(opinions=opinions):
                text_opinions.try_add_linked_text_opinions(
                    linked_text_opinions=linked_wrap,
                    check_opinion_correctness=lambda text_opinion: __check_text_opinion(text_opinion, terms_per_context))

    return text_opinions
