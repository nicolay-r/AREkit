import logging

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.ruattitudes.collection import RuAttitudesCollection
from arekit.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.source.ruattitudes.news.base import RuAttitudesNews


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    stemmer = MystemWrapper()

    ids = set()

    news_it = RuAttitudesCollection.iter_news(stemmer=stemmer,
                                              version=RuAttitudesVersions.V20,
                                              get_news_index_func=lambda: len(ids))

    for news in news_it:
        assert(isinstance(news, RuAttitudesNews))
        if news.ID in ids:
            logging.debug("index already exist: {}".format(news.ID))
        ids.add(news.ID)

    logger.debug("OK")
