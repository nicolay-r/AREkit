class BaseDocumentStatGenerator(object):
    """
    Provides statistic on certain document.
    Abstract, considered a specific implementation for document processing operation.
    """

    def __init__(self, doc_reader_func):
        """
        news_parser_func: func -> news
            assumes to provide a news by a certain doc_id
        """
        assert(callable(doc_reader_func))
        self.__doc_reader_func = doc_reader_func

    # region abstract protected methods

    def _calc(self, news):
        """ Abstract method that provides quantitative statistic
            for a particular news
        """
        raise NotImplementedError()

    # endregion

    # region public methods

    def calculate(self, doc_ids_iter):
        docs_info = []

        for doc_id in doc_ids_iter:
            doc = self.__doc_reader_func(doc_id)
            s_count = self._calc(doc)
            docs_info.append((doc_id, s_count))

        return docs_info

    # endregion
