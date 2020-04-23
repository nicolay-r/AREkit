from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.contrib.experiments.single.helpers.parsed_news import ParsedNewsHelper
from arekit.contrib.networks.sample import InputSample
from arekit.networks.context.debug import DebugKeys
from arekit.common.data_type import DataType
from arekit.source.rusentiframes.helpers.parse import RuSentiFramesParseHelper


# region private methods

# TODO. Useless method, could be removed.
def __read_document(experiment_io, doc_id):
    assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))
    assert(isinstance(doc_id, int))

    news, parsed_news = experiment_io.read_parsed_news(doc_id=doc_id)

    if DebugKeys.NewsTermsStatisticShow:
        ParsedNewsHelper.debug_statistics(parsed_news)
    if DebugKeys.NewsTermsShow:
        ParsedNewsHelper.debug_show_terms(parsed_news)

    return news, parsed_news


def __iter_opinion_collections(experiment_io, news_id, data_type):
    assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))
    assert(isinstance(news_id, int))
    assert(isinstance(data_type, unicode))

    neutral = experiment_io.read_neutral_opinion_collection(doc_id=news_id,
                                                            data_type=data_type)

    if neutral is not None:
        yield neutral

    if data_type == DataType.Train:
        yield experiment_io.read_etalon_opinion_collection(doc_id=news_id)


# TODO. This should be public
def __check_text_opinion(text_opinion, terms_per_context):
    assert(isinstance(text_opinion, TextOpinion))
    return InputSample.check_ability_to_create_sample(
        window_size=terms_per_context,
        text_opinion=text_opinion)

# endregions


def extract_text_opinions(experiment_io,
                          data_type,
                          terms_per_context):
    assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))
    assert(isinstance(data_type, unicode))
    assert(isinstance(terms_per_context, int))
    assert(terms_per_context > 0)

    parsed_collection = ParsedNewsCollection()

    text_opinions = LabeledLinkedTextOpinionCollection(
        parsed_news_collection=parsed_collection)

    for news_id in experiment_io.iter_news_indices(data_type):

        news, parsed_news = __read_document(experiment_io=experiment_io, doc_id=news_id)

        parsed_news.modify_parsed_sentences(
            lambda sentence: RuSentiFramesParseHelper.parse_frames_in_parsed_text(
                frame_variants_collection=experiment_io.DataIO.FrameVariantCollection,
                parsed_text=sentence))

        if not parsed_collection.contains_id(news_id):
            parsed_collection.add(parsed_news)
        else:
            print "Warning: Skipping document with id={}, news={}".format(news.DocumentID, news)

        opinions_it = __iter_opinion_collections(experiment_io=experiment_io,
                                                 news_id=news_id,
                                                 data_type=data_type)

        for opinions in opinions_it:
            for linked_text_opinions in news.iter_linked_text_opinions(opinions=opinions):
                text_opinions.try_add_linked_text_opinions(
                    linked_text_opinions=linked_text_opinions,
                    check_opinion_correctness=lambda text_opinion: __check_text_opinion(text_opinion, terms_per_context))

    return text_opinions
