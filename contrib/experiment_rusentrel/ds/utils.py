import logging

from arekit.common.utils import progress_bar_iter
from arekit.contrib.experiment_rusentrel.labels.scalers.ruattitudes import ExperimentRuAttitudesLabelScaler
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news.base import RuAttitudesNews


logger = logging.getLogger(__name__)


def read_ruattitudes_in_memory(version, keep_doc_ids_only, used_doc_ids_set=None):
    """
    Performs reading of ruattitude formatted documents and
    selection according to 'doc_ids_set' parameter.

    used_doc_ids_set: set or None
        ids of documents that already used and could not be assigned
        'None' corresponds to an empty set.
    """
    assert(isinstance(version, RuAttitudesVersions))
    assert(isinstance(keep_doc_ids_only, bool))
    assert(isinstance(used_doc_ids_set, set) or used_doc_ids_set is None)

    d = {}
    id_offset = max(used_doc_ids_set) + 1 if used_doc_ids_set is not None else 0

    it = RuAttitudesCollection.iter_news(version=version,
                                         get_news_index_func=lambda _: id_offset + len(d),
                                         label_scaler=ExperimentRuAttitudesLabelScaler(),
                                         return_inds_only=keep_doc_ids_only)

    it_formatted_and_logged = progress_bar_iter(
        iterable=__iter_id_with_news(news_it=it,
                                     keep_doc_ids_only=keep_doc_ids_only),
        desc=u"Loading RuAttitudes Collection [{}]".format(u"doc ids only" if keep_doc_ids_only else u"fully"),
        unit=u'docs')

    for news_id, news in it_formatted_and_logged:
        if used_doc_ids_set is not None:
            if news_id in used_doc_ids_set:
                logger.info(u"Document with id='{}' already used. Skipping".format(news_id))
                continue

        d[news_id] = news

    return d


def __iter_id_with_news(news_it, keep_doc_ids_only):
    if keep_doc_ids_only:
        for news_id in news_it:
            yield news_id, None
    else:
        for news in news_it:
            assert (isinstance(news, RuAttitudesNews))
            yield news.ID, news
