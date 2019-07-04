from core.common.bound import Bound
from core.common.parsed_news.parsed_news import ParsedNews
from core.processing.lemmatization.base import Stemmer
from core.processing.text.parser import TextParser
from core.source.rusentrel.entities.entity import RuSentRelEntity
from core.source.rusentrel.news import RuSentRelNews
from core.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelParsedNews(ParsedNews):

    @classmethod
    def create_from_rusentrel_news(cls, rusentrel_news_id, rusentrel_news, stemmer, keep_tokens):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(rusentrel_news, RuSentRelNews))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(keep_tokens, bool))

        parsed_sentences_iter = cls.__to_parsed_sentences(rusentrel_news=rusentrel_news,
                                                          keep_tokens=keep_tokens,
                                                          stemmer=stemmer)

        return cls(news_id=rusentrel_news_id,
                   parsed_sentences=list(parsed_sentences_iter))

    @staticmethod
    def __to_parsed_sentences(rusentrel_news, keep_tokens, stemmer):
        assert(isinstance(rusentrel_news, RuSentRelNews))
        assert(isinstance(keep_tokens, bool))
        assert(isinstance(stemmer, Stemmer))

        for s_index, sentence in enumerate(rusentrel_news.iter_sentences()):
            string_list = RuSentRelParsedNews.__sentence_to_list(sentence)
            parsed_sentence = TextParser.parse_string_list(string_list,
                                                           keep_tokens=keep_tokens,
                                                           stemmer=stemmer)
            yield parsed_sentence

    @staticmethod
    def __sentence_to_list(sentence):
        assert(isinstance(sentence, RuSentRelSentence))

        start = 0
        string_list = []
        for entity, bound in sentence.iter_entity_with_local_bounds():
            assert(isinstance(entity, RuSentRelEntity))
            assert(isinstance(bound, Bound))
            string_list.append(sentence.Text[start:bound.Position - start])
            string_list.append(entity)
            start = bound.Position + bound.Length

        string_list.append(sentence.Text[start:len(sentence.Text) - start])
        return string_list
