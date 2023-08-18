from arekit.common.data.doc_provider import DocumentProvider
from arekit.contrib.source.nerelbio.reader import NerelBioDocReader
from arekit.contrib.source.nerelbio.versions import NerelBioVersions


class NERELBioDocProvider(DocumentProvider):
    """ NEREL-BIO extends the general domain dataset NEREL.
        NEREL-BIO annotation scheme covers both general and biomedical
        domains making it suitable for domain transfer experiments.
        https://github.com/nerel-ds/NEREL-BIO
    """

    def __init__(self, filename_by_id, version):
        """ filename_ids: dict
                Dictionary of {id: filename}, where
                    - id: int
                    - filename: str
            version: NerelBioVersions
                Specify the appropriate version of the NEREL-BIO collection.
        """
        assert(isinstance(filename_by_id, dict))
        assert(isinstance(version, NerelBioVersions))
        super(NERELBioDocProvider, self).__init__()
        self.__filename_by_id = filename_by_id
        self.__version = version
        self.__doc_reader = NerelBioDocReader(version)

    def by_id(self, doc_id):
        return self.__doc_reader.read_document(doc_id=doc_id, filename=self.__filename_by_id[doc_id])
