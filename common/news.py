from arekit.common.entities.base import Entity
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.frame_variants.parse import FrameVariantsParser
from arekit.common.parsed_news.base import ParsedNews
from arekit.processing.text.token import Token


class News(object):

    NewsTermsShow = False
    NewsTermsStatisticShow = False

    def __init__(self, news_id):
        assert(isinstance(news_id, int))
        self.__news_id = news_id

    @property
    def ID(self):
        return self.__news_id

    def iter_wrapped_linked_text_opinions(self, opinions):
        """
        opinions: iterable Opinion
            is an iterable opinions that should be used to find a related text_opinion entries.
        """
        raise NotImplementedError()

    def parse(self, options, frame_variant_collection):
        assert(isinstance(frame_variant_collection, FrameVariantsCollection) or frame_variant_collection is None)

        parsed_news = self._parse_core(options)

        if self.NewsTermsStatisticShow:
            self.__debug_statistics(parsed_news)
        if self.NewsTermsShow:
            self.__debug_show_terms(parsed_news)

        self.__post_processing(parsed_news=parsed_news,
                               frame_variant_collection=frame_variant_collection)

        return parsed_news

    def _parse_core(self, options):
        """
        returns: ParsedNews
        """
        raise NotImplementedError()

    # region private methods

    def __post_processing(self, parsed_news, frame_variant_collection):
        """
        Labeling frame variants in doc sentences.
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(frame_variant_collection, FrameVariantsCollection) or frame_variant_collection is None)

        if frame_variant_collection is None:
            return

        parsed_news.modify_parsed_sentences(
            lambda sentence: FrameVariantsParser.parse_frames_in_parsed_text(
                frame_variants_collection=frame_variant_collection,
                parsed_text=sentence))

    @staticmethod
    def __debug_show_terms(parsed_news):
        assert(isinstance(parsed_news, ParsedNews))
        for term in parsed_news.iter_terms():
            if isinstance(term, unicode):
                print "Word:    '{}'".format(term.encode('utf-8'))
            elif isinstance(term, Token):
                print "Token:   '{}' ('{}')".format(term.get_token_value().encode('utf-8'),
                                                    term.get_meta_value().encode('utf-8'))
            elif isinstance(term, Entity):
                print "Entity:  '{}'".format(term.Value.encode('utf-8'))
            else:
                raise Exception("unsuported type {}".format(term))

    @staticmethod
    def __debug_statistics(parsed_news):
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

    # endregion
