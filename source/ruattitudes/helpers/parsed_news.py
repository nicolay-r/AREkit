from core.common.parsed_news.parsed_news import ParsedNews
from core.source.ruattitudes.news import RuAttitudesNews
from core.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesParsedNewsHelper:

    @classmethod
    def create_parsed_news(cls, doc_id, news):
        assert(isinstance(doc_id, int))
        assert(isinstance(news, RuAttitudesNews))

        parsed_sentences_iter = cls.__iter_as_sentences_with_entities(news)

        return ParsedNews(news_id=doc_id,
                          parsed_sentences=parsed_sentences_iter)

    @staticmethod
    def __iter_as_sentences_with_entities(news):
        """
        This method returns sentences with labeled entities in it.
        """
        assert(isinstance(news, RuAttitudesNews))

        for sentence in news.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))

            # TODO. label objects in sentence.
            parsed_text = sentence.ParsedText

            # TODO. Implement

            yield parsed_text
