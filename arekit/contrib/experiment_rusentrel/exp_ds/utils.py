import logging

from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news.base import RuAttitudesNews


logger = logging.getLogger(__name__)


def read_ruattitudes_in_memory(version, keep_doc_ids_only, doc_id_func):
    """ Performs reading of ruattitude formatted documents and
        selection according to 'doc_ids_set' parameter.
    """
    assert(isinstance(version, RuAttitudesVersions))
    assert(isinstance(keep_doc_ids_only, bool))
    assert(callable(doc_id_func))

    it = RuAttitudesCollection.iter_news(version=version,
                                         get_news_index_func=doc_id_func,
                                         return_inds_only=keep_doc_ids_only)

    it_formatted_and_logged = progress_bar_iter(
        iterable=__iter_id_with_news(news_it=it,
                                     keep_doc_ids_only=keep_doc_ids_only),
        desc="Loading RuAttitudes Collection [{}]".format("doc ids only" if keep_doc_ids_only else "fully"),
        unit='docs')

    d = {}
    for doc_id, news in it_formatted_and_logged:
        d[doc_id] = news

    return d


def __iter_id_with_news(news_it, keep_doc_ids_only):
    if keep_doc_ids_only:
        for doc_id in news_it:
            yield doc_id, None
    else:
        for news in news_it:
            assert (isinstance(news, RuAttitudesNews))
            yield news.ID, news
