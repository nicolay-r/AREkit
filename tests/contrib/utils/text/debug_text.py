import logging

from arekit.common.context.token import Token
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.docs.parsed.base import ParsedDocument


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


def debug_show_doc_terms(parsed_doc):
    assert(isinstance(parsed_doc, ParsedDocument))
    logger.info('------------------------')
    return debug_show_terms(terms=parsed_doc.iter_terms())


def debug_show_terms(terms):
    for term in terms:
        if isinstance(term, str):
            logger.debug("Word:\t\t'{}'".format(term))
        elif isinstance(term, Token):
            logger.debug("Token:\t\t'{}' ('{}')".format(term.get_token_value(),
                                                        term.get_meta_value()))
        elif isinstance(term, Entity):
            logger.debug("Entity:\t\t'{}'".format(term.Value))
        elif isinstance(term, TextFrameVariant):
            text = "TextFV({is_neg}):\t'{v}'".format(is_neg="+" if term.IsNegated else "-",
                                                     v=term.Variant.get_value())
            logger.debug(text)
        else:
            raise Exception("unsuported type {}".format(term))