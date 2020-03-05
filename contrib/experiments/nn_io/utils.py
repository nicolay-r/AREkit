from arekit.processing.lemmatization.base import Stemmer
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.ruattitudes.reader import RuAttitudesFormatReader


def read_ruattitudes_in_memory(stemmer, doc_ids_set=None):
    """
    Performs reading of ruattitude formatted documents and
    selection according to 'doc_ids_set' parameter.

    doc_ids_set: set or None
        ids of documents that should be selected.
        'None' corresponds to all the available doc_ids.
    """
    assert(isinstance(stemmer, Stemmer))
    assert(isinstance(doc_ids_set, set) or doc_ids_set is None)

    d = {}

    for news in RuAttitudesFormatReader.iter_news(stemmer=stemmer):
        assert(isinstance(news, RuAttitudesNews))

        if doc_ids_set is not None and news.NewsIndex not in doc_ids_set:
            continue

        d[news.NewsIndex] = news

    return d
