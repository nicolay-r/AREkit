from arekit.processing.text.parser import TextParser


class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def iter_supported_data_types(self):
        """ Iterates through data_types, supported in a related experiment
            Note:
            In CV-split algorithm, the first part corresponds to a LARGE split,
            Jand second to small; therefore, the correct sequence is as follows:
            DataType.Train, DataType.Test.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def iter_parsed_news(self, doc_inds):
        for doc_id in doc_inds:
            yield self.__parse_news(doc_id=doc_id)

    def _create_parse_options(self):
        raise NotImplementedError()

    def __parse_news(self, doc_id):
        news = self.read_news(doc_id=doc_id)
        return TextParser.parse_news(news=news,
                                     parse_options=self._create_parse_options())
