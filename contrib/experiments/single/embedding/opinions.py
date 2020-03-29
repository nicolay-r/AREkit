from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.contrib.experiments.operations.opinions import OpinionOperations
from arekit.contrib.experiments.single.helpers.parsed_news import ParsedNewsHelper
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample
from arekit.networks.context.debug import DebugKeys
from arekit.networks.data_type import DataType
from arekit.source.ruattitudes.helpers.linked_text_opinions import RuAttitudesNewsTextOpinionExtractorHelper
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.rusentrel.helpers.linked_text_opinions import RuSentRelNewsTextOpinionExtractorHelper
from arekit.source.rusentiframes.helpers.parse import RuSentiFramesParseHelper
from arekit.source.rusentrel.news import RuSentRelNews


# region private methods

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


def __iter_opinion_collections(opinion_operations, news_id, data_type):
    assert(isinstance(opinion_operations, OpinionOperations))
    assert(isinstance(news_id, int))
    assert(isinstance(data_type, unicode))

    neutral = opinion_operations.read_neutral_opinion_collection(doc_id=news_id,
                                                                 data_type=data_type)

    if neutral is not None:
        yield neutral

    if data_type == DataType.Train:
        yield opinion_operations.read_etalon_opinion_collection(doc_id=news_id)


def __fill_text_opinions(text_opinions,
                         news,
                         opinions,
                         terms_per_context):
    assert(isinstance(news, RuSentRelNews) or isinstance(news, RuAttitudesNews))
    assert(isinstance(text_opinions, LabeledLinkedTextOpinionCollection))
    assert(isinstance(opinions, OpinionCollection))
    assert(isinstance(terms_per_context, int))

    def __check_text_opinion(text_opinion):
        assert(isinstance(text_opinion, TextOpinion))
        return InputSample.check_ability_to_create_sample(
            window_size=terms_per_context,
            text_opinion=text_opinion)

    # TODO. Here is a dependency from certain format.
    # TODO. Add entries duplicated in different classes.
    # TODO. Here only an iterator.
    # TODO. Use  collection filling (text_opinions) later.

    if isinstance(news, RuSentRelNews):
        return RuSentRelNewsTextOpinionExtractorHelper.add_entries(
            text_opinion_collection=text_opinions,
            news=news,
            opinions=opinions,
            check_text_opinion_is_correct=__check_text_opinion)

    elif isinstance(news, RuAttitudesNews):
        return RuAttitudesNewsTextOpinionExtractorHelper.add_entries(
            text_opinion_collection=text_opinions,
            news=news,
            check_text_opinion_is_correct=__check_text_opinion)

    # TODO. Fill here.

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

        opinions_it = __iter_opinion_collections(opinion_operations=experiment_io,
                                                 news_id=news_id,
                                                 data_type=data_type)

        for opinions in opinions_it:
            __fill_text_opinions(text_opinions=text_opinions,
                                 news=news,
                                 opinions=opinions,
                                 terms_per_context=config.TermsPerContext)

    return text_opinions
