from arekit.common import utils
from arekit.common.parsed_news.base import ParsedNews
from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.parser import TextParser
from arekit.source.rusentrel.news import RuSentRelNews


class RuSentRelParsedNewsHelper:

    @staticmethod
    def create_parsed_news(rusentrel_news_id, rusentrel_news, keep_tokens, stemmer=None):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(rusentrel_news, RuSentRelNews))
        assert(isinstance(stemmer, Stemmer) or isinstance(stemmer, type(None)))
        assert(isinstance(keep_tokens, bool))

        parsed_sentences_iter = RuSentRelParsedNewsHelper.__iter_parsed_sentences(
            rusentrel_news=rusentrel_news,
            keep_tokens=keep_tokens,
            stemmer=stemmer)

        return ParsedNews(news_id=rusentrel_news_id,
                          parsed_sentences=parsed_sentences_iter)

    # region private methods

    @staticmethod
    def __iter_parsed_sentences(rusentrel_news, keep_tokens, stemmer):
        assert(isinstance(rusentrel_news, RuSentRelNews))
        assert(isinstance(keep_tokens, bool))
        assert(isinstance(stemmer, Stemmer) or isinstance(stemmer, type(None)))

        for s_index, sentence in enumerate(rusentrel_news.iter_sentences()):

            string_iter = utils.iter_text_with_substitutions(
                text=sentence.Text,
                iter_subs=sentence.iter_entity_with_local_bounds())

            yield TextParser.parse_string_list(string_iter=string_iter,
                                               keep_tokens=keep_tokens,
                                               stemmer=stemmer)

    # endregion
