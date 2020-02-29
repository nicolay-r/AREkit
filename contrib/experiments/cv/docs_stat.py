class DocStatGeneratorBase(object):
    """
    Provides statistic on certain document.
    Abstract, considered a specific implementation for document processing operation.
    """

    def calculate_sentences_count(self, doc_id):
        raise NotImplementedError()

    def iter_doc_ids(self):
        raise NotImplementedError()

    # region public methods

    def write_doc_stat(self, filepath):
        with open(filepath, 'w') as f:
            for doc_index in self.iter_doc_ids():
                s_count = self.calculate_sentences_count(doc_index)
                f.write("{}: {}\n".format(doc_index, s_count))

    def read_docs_stat(self, filepath):
        """
        return:
            list of the following pairs: (doc_id, sentences_count)
        """
        docs_info = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                args = [int(i) for i in line.split(':')]
                doc_id, s_count = args
                docs_info.append((doc_id, s_count))

        return docs_info

    # endregion
