from os.path import join

from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.sample import InputSample


def get_path_of_subfolder_in_experiments_dir(subfolder_name, experiments_dir):
    """
    Returns subfolder in experiments directory
    """
    assert(isinstance(subfolder_name, unicode))
    assert(isinstance(experiments_dir, unicode))

    target_dir = join(experiments_dir, u"{}/".format(subfolder_name))
    create_dir_if_not_exists(target_dir)
    return target_dir


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

    return InputSample.from_text_opinion(
        text_opinion=text_opinion,
        parsed_news=parsed_news_collection.get_by_news_id(text_opinion.NewsID),
        config=config,
        frames_collection=frames_collection,
        synonyms_collection=synonyms_collection)

