import logging

from arekit.common.entities.base import Entity
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.processing.text.token import Token


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


def debug_show_terms(terms):
    for term in terms:
        if isinstance(term, unicode):
            logger.debug("Word:\t\t'{}'".format(term.encode('utf-8')))
        elif isinstance(term, Token):
            logger.debug("Token:\t\t'{}' ('{}')".format(term.get_token_value().encode('utf-8'),
                                                       term.get_meta_value().encode('utf-8')))
        elif isinstance(term, Entity):
            logger.debug("Entity:\t\t'{}'".format(term.Value.encode('utf-8')))
        elif isinstance(term, TextFrameVariant):
            logger.debug("TextFV:\t\t'{}'".format(term.Variant.get_value().encode('utf-8')))
        else:
            raise Exception("unsuported type {}".format(term))


def debug_show_news_terms(parsed_news):
    assert(isinstance(parsed_news, ParsedNews))
    logger.info('------------------------')
    return debug_show_terms(terms=parsed_news.iter_terms())


def debug_statistics(parsed_news):
    assert(isinstance(parsed_news, ParsedNews))

    terms = list(parsed_news.iter_terms())
    words = filter(lambda term: isinstance(term, unicode), terms)
    tokens = filter(lambda term: isinstance(term, Token), terms)
    entities = filter(lambda term: isinstance(term, Entity), terms)
    total = len(words) + len(tokens) + len(entities)

    logger.debug("Extracted news_words info, NEWS_ID: {}".format(parsed_news.RelatedNewsID))
    logger.debug("\tWords: {} ({}%)".format(len(words), 100.0 * len(words) / total))
    logger.debug("\tTokens: {} ({}%)".format(len(tokens), 100.0 * len(tokens) / total))
    logger.debug("\tEntities: {} ({}%)".format(len(entities), 100.0 * len(entities) / total))

