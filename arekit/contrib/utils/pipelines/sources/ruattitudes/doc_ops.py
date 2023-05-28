from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news import RuAttitudesNews
from arekit.contrib.source.ruattitudes.news_brat import RuAttitudesNewsConverter
from arekit.contrib.utils.data.doc_ops.dict_based import DictionaryBasedDocumentOperations


class RuAttitudesDocumentOperations(DictionaryBasedDocumentOperations):

    def __init__(self, version, keep_doc_ids_only, doc_id_func, limit):
        d = self.read_ruattitudes_to_brat_in_memory(version=version,
                                                    keep_doc_ids_only=keep_doc_ids_only,
                                                    doc_id_func=doc_id_func,
                                                    limit=limit)
        super(RuAttitudesDocumentOperations, self).__init__(d)

    @staticmethod
    def read_ruattitudes_to_brat_in_memory(version, keep_doc_ids_only, doc_id_func, limit=None):
        """ Performs reading of RuAttitude formatted documents and
            selection according to 'doc_ids_set' parameter.
        """
        assert (isinstance(version, RuAttitudesVersions))
        assert (isinstance(keep_doc_ids_only, bool))
        assert (callable(doc_id_func))

        it = RuAttitudesCollection.iter_news(version=version,
                                             get_news_index_func=doc_id_func,
                                             return_inds_only=keep_doc_ids_only)

        it_formatted_and_logged = progress_bar_iter(
            iterable=RuAttitudesDocumentOperations.__iter_id_with_news(
                docs_it=it, keep_doc_ids_only=keep_doc_ids_only),
            desc="Loading RuAttitudes Collection [{}]".format("doc ids only" if keep_doc_ids_only else "fully"),
            unit='docs')

        d = {}
        docs_read = 0
        for doc_id, news in it_formatted_and_logged:
            assert(isinstance(news, RuAttitudesNews) or news is None)
            d[doc_id] = RuAttitudesNewsConverter.to_brat_news(news) if news is not None else None
            docs_read += 1
            if limit is not None and docs_read >= limit:
                break

        return d

    @staticmethod
    def __iter_id_with_news(docs_it, keep_doc_ids_only):
        if keep_doc_ids_only:
            for doc_id in docs_it:
                yield doc_id, None
        else:
            for news in docs_it:
                assert (isinstance(news, RuAttitudesNews))
                yield news.ID, news
