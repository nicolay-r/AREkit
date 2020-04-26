from arekit.common.entities.base import Entity
from arekit.common.parsed_news.base import ParsedNews
from arekit.processing.text.token import Token


class ParsedNewsHelper(object):

    @staticmethod
    def debug_show_terms(parsed_news):
        assert(isinstance(parsed_news, ParsedNews))
        for term in parsed_news.iter_terms():
            if isinstance(term, unicode):
                print "Word:    '{}'".format(term.encode('utf-8'))
            elif isinstance(term, Token):
                print "Token:   '{}' ('{}')".format(term.get_token_value().encode('utf-8'),
                                                    term.get_original_value().encode('utf-8'))
            elif isinstance(term, Entity):
                print "Entity:  '{}'".format(term.Value.encode('utf-8'))
            else:
                raise Exception("unsuported type {}".format(term))

    @staticmethod
    def debug_statistics(parsed_news):
        assert(isinstance(parsed_news, ParsedNews))

        terms = list(parsed_news.iter_terms())
        words = filter(lambda term: isinstance(term, unicode), terms)
        tokens = filter(lambda term: isinstance(term, Token), terms)
        entities = filter(lambda term: isinstance(term, Entity), terms)
        total = len(words) + len(tokens) + len(entities)

        print "Extracted news_words info, NEWS_ID: {}".format(parsed_news.RelatedNewsID)
        print "\tWords: {} ({}%)".format(len(words), 100.0 * len(words) / total)
        print "\tTokens: {} ({}%)".format(len(tokens), 100.0 * len(tokens) / total)
        print "\tEntities: {} ({}%)".format(len(entities), 100.0 * len(entities) / total)


