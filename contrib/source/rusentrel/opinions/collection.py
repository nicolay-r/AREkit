from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter


class RuSentRelOpinionCollection:
    """
    Collection of sentiment opinions between entities
    """

    # TODO. Provide here an opportunity with synonyms=None
    @staticmethod
    def load_collection(doc_id, synonyms=None, version=RuSentRelVersions.V11):
        """
        doc_id:
        synonyms: None or SynonymsCollection
            None corresponds to the related synonym collection from RuSentRel collection.
        version:
        """
        assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)
        assert(isinstance(version, RuSentRelVersions))

        use_native_collection = synonyms is None

        if use_native_collection:
            # TODO. Now it is not supported, since synonyms collection
            # TODO. requires to use stemmer in initialization.
            synonyms = None

        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_sentiment_opin_filepath(doc_id),
            process_func=lambda input_file: RuSentRelOpinionCollectionFormatter._load_from_file(
                input_file=input_file,
                synonyms=synonyms,
                labels_formatter=RuSentRelLabelsFormatter(),
                is_native_synonyms_collection=use_native_collection),
            version=version)
