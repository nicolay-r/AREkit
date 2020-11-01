from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter


class RuSentRelOpinionCollection:
    """
    Collection of sentiment opinions between entities
    """

    @staticmethod
    def iter_opinions_from_doc(doc_id, version=RuSentRelVersions.V11):
        """
        doc_id:
        synonyms: None or SynonymsCollection
            None corresponds to the related synonym collection from RuSentRel collection.
        version:
        """
        assert(isinstance(version, RuSentRelVersions))

        labels_fmt = RuSentRelLabelsFormatter()

        return RuSentRelIOUtils.iter_from_zip(
            inner_path=RuSentRelIOUtils.get_sentiment_opin_filepath(doc_id),
            process_func=lambda input_file: RuSentRelOpinionCollectionFormatter._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=labels_fmt),
            version=version)
