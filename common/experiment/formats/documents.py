from arekit.common.experiment.folding.base import BaseExperimentDataFolding
from arekit.processing.text.parser import TextParser


class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def __init__(self, folding):
        assert(isinstance(folding, BaseExperimentDataFolding))
        self.__folding = folding

    @property
    def DataFolding(self):
        """ Algorithm, utilized in order to provide variety of foldings, such as
                cross-validation, fixed, none, etc.
        """
        return self.__folding

    # TODO. Into Neutral annotator!!!
    # TODO. Into Neutral annotator!!!
    # TODO. Into Neutral annotator!!!
    def get_doc_ids_set_to_neutrally_annotate(self):
        """ provides set of documents that utilized by neutral annotator algorithm in order to
            provide the related labeling of neutral attitudes in it.
            By default we consider an empty set, so there is no need to ulize neutral annotator.
        """
        raise NotImplementedError()

    def read_news(self, doc_id):
        raise NotImplementedError()

    def iter_news_indices(self, data_type):
        """ Provides a news indeces, related to a particular `data_type`
        """
        data_types_splits = self.__folding.fold_doc_ids_set()
        for doc_id in data_types_splits[data_type]:
            yield doc_id

    def iter_parsed_news(self, doc_inds):
        for doc_id in doc_inds:
            yield self.__parse_news(doc_id=doc_id)

    def _create_parse_options(self):
        raise NotImplementedError()

    def __parse_news(self, doc_id):
        news = self.read_news(doc_id=doc_id)
        return TextParser.parse_news(news=news,
                                     parse_options=self._create_parse_options())
