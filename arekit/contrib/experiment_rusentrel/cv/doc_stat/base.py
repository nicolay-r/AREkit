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

    # TODO. depends on io, issue #189
    def calculate_and_write_doc_stat(self, filepath, doc_ids_iter):
        with open(filepath, 'w') as f:
            for doc_id in doc_ids_iter:
                doc = self.__doc_reader_func(doc_id)
                s_count = self._calc(doc)
                f.write("{}: {}\n".format(doc_id, s_count))

    @staticmethod
    def read_docs_stat(filepath, doc_ids_set):
        """
        doc_ids_set: set
            set of documents expected to be extracted

        return:
            list of the following pairs: (doc_id, sentences_count)
        """
        assert(isinstance(doc_ids_set, set))

        docs_info = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                args = [int(i) for i in line.split(':')]
                doc_id, s_count = args

                if doc_id not in doc_ids_set:
                    continue

                docs_info.append((doc_id, s_count))

        return docs_info

    # endregion
