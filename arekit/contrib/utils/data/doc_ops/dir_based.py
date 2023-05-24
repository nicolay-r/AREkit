from os.path import join

from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.news.base import News
from arekit.common.news.sentence import BaseNewsSentence


class DirectoryFilesDocOperations(DocumentOperations):
    """ Document Operations based on the list of provided file paths
        for the particular directory.
    """

    def __init__(self, dir_path, file_names=None, sentence_parser=None):
        """
            dir_path: str
                path to the root directory.
            file_names: list
                list of file paths related to documents.
            sentence_splitter: object
                how data is suppose to be separated onto sentences.
                str -> list(str)
        """
        assert(isinstance(dir_path, str))
        assert(isinstance(file_names, list) or file_names is None)
        assert(callable(sentence_parser) or sentence_parser is None)

        self.__dir_path = dir_path
        self.__file_names = file_names

        # Line-split sentence parser by default.
        self.__sentence_parser = (lambda text: [t.strip() for t in text.split('\n')]) \
            if sentence_parser is None else sentence_parser

    def __read_doc(self, doc_id, contents):
        """ Parse a single document.
        """
        # setup input data.
        sentences = self.__sentence_parser(contents)
        sentences = list(map(lambda text: BaseNewsSentence(text), sentences))

        # Parse text.
        return News(doc_id=doc_id, sentences=sentences)

    def by_id(self, doc_id):
        """ Perform reading operation of the document.
        """
        file_name = self.__file_names[doc_id]
        with open(join(self.__dir_path, file_name), "r") as f:
            contents = f.read()
            return self.__read_doc(doc_id=file_name, contents=contents)

    def __len__(self):
        return len(self.__file_names)
