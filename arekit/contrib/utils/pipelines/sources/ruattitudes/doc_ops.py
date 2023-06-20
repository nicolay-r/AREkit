from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.doc import RuAttitudesDocument
from arekit.contrib.source.ruattitudes.doc_brat import RuAttitudesDocumentsConverter
from arekit.contrib.utils.data.doc_provider.dict_based import DictionaryBasedDocumentProvider


class RuAttitudesDocumentProvider(DictionaryBasedDocumentProvider):

    def __init__(self, version, keep_doc_ids_only, doc_id_func, limit):
        d = self.read_ruattitudes_to_brat_in_memory(version=version,
                                                    keep_doc_ids_only=keep_doc_ids_only,
                                                    doc_id_func=doc_id_func,
                                                    limit=limit)
        super(RuAttitudesDocumentProvider, self).__init__(d)

    @staticmethod
    def read_ruattitudes_to_brat_in_memory(version, keep_doc_ids_only, doc_id_func, limit=None):
        """ Performs reading of RuAttitude formatted documents and
            selection according to 'doc_ids_set' parameter.
        """
        assert (isinstance(version, RuAttitudesVersions))
        assert (isinstance(keep_doc_ids_only, bool))
        assert (callable(doc_id_func))

        it = RuAttitudesCollection.iter_docs(version=version,
                                             get_doc_index_func=doc_id_func,
                                             return_inds_only=keep_doc_ids_only)

        it_formatted_and_logged = progress_bar_iter(
            iterable=RuAttitudesDocumentProvider.__iter_id_with_doc(
                docs_it=it, keep_doc_ids_only=keep_doc_ids_only),
            desc="Loading RuAttitudes Collection [{}]".format("doc ids only" if keep_doc_ids_only else "fully"),
            unit='docs')

        d = {}
        docs_read = 0
        for doc_id, doc in it_formatted_and_logged:
            assert(isinstance(doc, RuAttitudesDocument) or doc is None)
            d[doc_id] = RuAttitudesDocumentsConverter.to_brat_doc(doc) if doc is not None else None
            docs_read += 1
            if limit is not None and docs_read >= limit:
                break

        return d

    @staticmethod
    def __iter_id_with_doc(docs_it, keep_doc_ids_only):
        if keep_doc_ids_only:
            for doc_id in docs_it:
                yield doc_id, None
        else:
            for doc in docs_it:
                assert (isinstance(doc, RuAttitudesDocument))
                yield doc.ID, doc
