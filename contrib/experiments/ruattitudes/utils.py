import logging

from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news.base import RuAttitudesNews


logger = logging.getLogger(__name__)


def read_ruattitudes_in_memory(version, used_doc_ids_set=None):
    """
    Performs reading of ruattitude formatted documents and
    selection according to 'doc_ids_set' parameter.

    used_doc_ids_set: set or None
        ids of documents that already used and could not be assigned
        'None' corresponds to an empty set.
    """
    assert (isinstance(version, RuAttitudesVersions))
    assert (isinstance(used_doc_ids_set, set) or used_doc_ids_set is None)

    id_offset = max(used_doc_ids_set) + 1 if used_doc_ids_set is not None else 0

    d = {}

    news_it = progress_bar_iter(
        iterable=RuAttitudesCollection.iter_news(
            version=version,
            get_news_index_func=lambda: id_offset + len(d)),
        desc=u"Loading RuAttitudes Collection",
        unit=u'docs')

    for news in news_it:
        assert (isinstance(news, RuAttitudesNews))

        if used_doc_ids_set is not None:
            if news.ID in used_doc_ids_set:
                logger.info(u"Document with id='{}' already used. Skipping".format(news.ID))
                continue

        d[news.ID] = news

    return d
