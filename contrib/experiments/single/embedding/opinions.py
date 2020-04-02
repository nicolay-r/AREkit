from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.contrib.experiments.single.helpers.parsed_news import ParsedNewsHelper
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample
from arekit.networks.context.debug import DebugKeys
from arekit.networks.data_type import DataType
from arekit.source.rusentiframes.helpers.parse import RuSentiFramesParseHelper


# region private methods

# TODO. Useless method, could be removed.
def __read_document(experiment_io, doc_id, config):
    assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))
    assert(isinstance(doc_id, int))
    assert(isinstance(config, DefaultNetworkConfig))

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
                          frame_variants_collection,
                          config):
    assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))
    assert(isinstance(data_type, unicode))
    assert(isinstance(config, DefaultNetworkConfig))
    assert(isinstance(frame_variants_collection, FrameVariantsCollection))

    parsed_collection = ParsedNewsCollection()

    text_opinions = LabeledLinkedTextOpinionCollection(
        parsed_news_collection=parsed_collection)

    for news_id in experiment_io.iter_news_indices(data_type):

        # TODO. method is useless, the code from the inside could be moved here.
        news, parsed_news = __read_document(experiment_io=experiment_io,
                                            doc_id=news_id,
                                            config=config)

        parsed_news.modify_parsed_sentences(
            lambda sentence: RuSentiFramesParseHelper.parse_frames_in_parsed_text(
                frame_variants_collection=frame_variants_collection,
                parsed_text=sentence))

        if not parsed_collection.contains_id(news_id):
            parsed_collection.add(parsed_news)
        else:
            print "Warning: Skipping document with id={}, news={}".format(news.DocumentID, news)

        opinions_it = __iter_opinion_collections(experiment_io=experiment_io,
                                                 news_id=news_id,
                                                 data_type=data_type)

        for opinions in opinions_it:
            text_opinions.try_add_linked_text_opinions(
                linked_text_opinions=news.iter_text_opinions(opinions=opinions),
                check_opinion_correctness=lambda text_opinion: __check_text_opinion(text_opinion, config.TermsPerContext))

    return text_opinions
