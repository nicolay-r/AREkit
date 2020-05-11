from arekit.common.linked.text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.text_opinions.helper import TextOpinionHelper
# TODO. This sample should be a parameter of a model.
from arekit.contrib.networks.sample import InputSample


# TODO. Provide experiment.
def create_input_sample(text_opinion, frames_collection, synonyms_collection, config):
    """
    Creates an input for Neural Network model
    """
    assert(isinstance(text_opinion, TextOpinion))
    assert(TextOpinionHelper.check_ends_has_same_sentence_index(text_opinion))

    text_opinion_collection = text_opinion.Owner
    assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))

    parsed_news_collection = text_opinion_collection.RelatedParsedNewsCollection
    assert(isinstance(parsed_news_collection, ParsedNewsCollection))

    # TODO. This sample should be a parameter of a model.
    return InputSample.from_text_opinion(
        text_opinion=text_opinion,
        parsed_news=parsed_news_collection.get_by_news_id(text_opinion.NewsID),
        config=config,
        frames_collection=frames_collection,
        synonyms_collection=synonyms_collection)
