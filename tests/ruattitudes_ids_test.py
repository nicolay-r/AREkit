import logging

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.ruattitudes.collection import RuAttitudesCollection
from arekit.source.ruattitudes.news.base import RuAttitudesNews


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    stemmer = MystemWrapper()

    ids = set()
    for news in RuAttitudesCollection.iter_news(stemmer):
        assert(isinstance(news, RuAttitudesNews))
        if news.ID in ids:
            logging.debug("index already exist: {}".format(news.ID))
        ids.add(news.ID)

    logger.debug("OK")
