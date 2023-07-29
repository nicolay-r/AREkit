from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.source.nerel.io_utils import NerelVersions
from arekit.contrib.source.nerel.reader import NerelDocReader


class NERELDocOperation(DocumentOperations):
    """ Document reader for the collection of the RuSentNE competition 2023.
        For more details please follow the following repository:
        github: https://github.com/dialogue-evaluation/RuSentNE-evaluation
    """

    def __init__(self, filename_by_id, version):
        """ filename_ids: dict
                Dictionary of {id: filename}, where
                    - id: int
                    - filename: str
            version: SentiNerelVersions
                Specify the appropriate version of teh SentiNEREL collection.
        """
        assert(isinstance(filename_by_id, dict))
        assert(isinstance(version, NerelVersions))
        super(NERELDocOperation, self).__init__()
        self.__filename_by_id = filename_by_id
        self.__version = version
        self.__doc_reader = NerelDocReader(version)

    def by_id(self, doc_id):
        return self.__doc_reader.read_document(doc_id=doc_id, filename=self.__filename_by_id[doc_id])
