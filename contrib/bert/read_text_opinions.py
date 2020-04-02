from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.contrib.networks.sample import InputSample
from arekit.networks.data_type import DataType
from arekit.source.ruattitudes.helpers.linked_text_opinions import RuAttitudesNewsTextOpinionExtractorHelper
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.rusentrel.helpers.linked_text_opinions import RuSentRelNewsTextOpinionExtractorHelper
from arekit.source.rusentrel.news import RuSentRelNews


def __iter_opinion_collections(io, news_id, data_type):
    """
    Iter collection from sources
    """
    assert(isinstance(news_id, int))
    assert(isinstance(data_type, unicode))

    neutral = io.read_neutral_opinion_collection(doc_id=news_id,
                                                 data_type=data_type)

    if neutral is not None:
        yield neutral

    if data_type == DataType.Train:
        yield io.read_etalon_opinion_collection(doc_id=news_id)


# TODO. Code is the same.
def __fill_text_opinions(text_opinions, news, opinions, terms_per_context):
    """
    Fill text_opinions collection
    """
    assert(isinstance(text_opinions, LabeledLinkedTextOpinionCollection))
    assert(isinstance(news, RuSentRelNews) or isinstance(news, RuAttitudesNews))
    assert(isinstance(opinions, OpinionCollection))
    assert(isinstance(terms_per_context, int))

    def __check_text_opinion(text_opinion):
        assert (isinstance(text_opinion, TextOpinion))
        return InputSample.check_ability_to_create_sample(
            window_size=terms_per_context,
            text_opinion=text_opinion)

    # For news from RuAttitudes collection
    if isinstance(news, RuSentRelNews):
        return RuSentRelNewsTextOpinionExtractorHelper.add_entries(
            text_opinion_collection=text_opinions,
            news=news,
            opinions=opinions,
            check_text_opinion_is_correct=__check_text_opinion)

    # For news from RuAttitudes collection
    elif isinstance(news, RuAttitudesNews):
        return RuAttitudesNewsTextOpinionExtractorHelper.add_entries(
            text_opinion_collection=text_opinions,
            news=news,
            check_text_opinion_is_correct=__check_text_opinion)


# TODO. This is the same and should be removed.
def extract_text_opinions(io, data_type, terms_per_context):
    assert(isinstance(io, BaseExperimentNeuralNetworkIO))
    assert(isinstance(data_type, unicode))
    assert(isinstance(terms_per_context, int))

    parsed_collection = ParsedNewsCollection()
    text_opinions = LabeledLinkedTextOpinionCollection(parsed_news_collection=parsed_collection)

    # TODO. Use data_type parameter instead
    news_ids = io.iter_test_data_indices() \
        if data_type == DataType.Test else \
        io.iter_train_data_indices()

    for news_id in news_ids:

        news, parsed_news = io.read_parsed_news(doc_id=news_id)

        if not parsed_collection.contains_id(news_id):
            parsed_collection.add(parsed_news)
        else:
            print "Warning: Skipping document with id={}, news={}".format(news.DocumentID, news)

        for opinions in __iter_opinion_collections(io=io, news_id=news_id, data_type=data_type):
            __fill_text_opinions(text_opinions=text_opinions,
                                 news=news,
                                 opinions=opinions,
                                 terms_per_context=terms_per_context)

    return text_opinions
