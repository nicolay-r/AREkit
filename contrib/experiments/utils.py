import glob
import os
import shutil

from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.contrib.networks.sample import InputSample


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

# TODO. To data_io
def rm_dir_contents(dir_path):
    contents = glob.glob(dir_path)
    for f in contents:
        print "Removing old file/dir: {}".format(f)
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f, ignore_errors=True)

# TODO. Rename this file as sample.