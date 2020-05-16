# -*- coding: utf-8 -*-
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter


class RuSentRelOpinionCollection:
    """
    Collection of sentiment opinions between entities
    """

    @staticmethod
    def load_collection(doc_id, synonyms, version=RuSentRelVersions.V11):
        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_sentiment_opin_filepath(doc_id),
            process_func=lambda input_file: RuSentRelOpinionCollectionFormatter._load_from_file(
                input_file=input_file,
                synonyms=synonyms,
                labels_formatter=RuSentRelLabelsFormatter()),
            version=version)
