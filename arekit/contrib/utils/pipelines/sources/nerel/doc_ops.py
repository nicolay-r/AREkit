from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.source.nerel.reader import NerelDocReader
from arekit.contrib.source.nerel.versions import NerelVersions


class NERELDocOperation(DocumentOperations):
    """ A Russian dataset with nested named entities, relations, events and linked entities.
        https://github.com/nerel-ds/NEREL
    """

    def __init__(self, filename_by_id, version):
        """ filename_ids: dict
                Dictionary of {id: filename}, where
                    - id: int
                    - filename: str
            version: NerelVersions
                Specify the appropriate version of teh NEREL collection.
        """
        assert(isinstance(filename_by_id, dict))
        assert(isinstance(version, NerelVersions))
        super(NERELDocOperation, self).__init__()
        self.__filename_by_id = filename_by_id
        self.__version = version
        self.__doc_reader = NerelDocReader(version)

    def by_id(self, doc_id):
        return self.__doc_reader.read_document(doc_id=doc_id, filename=self.__filename_by_id[doc_id])
