from arekit.common.news.base import News
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.text.parser import BaseTextParser


class NewsParser(object):

    @staticmethod
    def parse(news, text_parser):
        assert(isinstance(news, News))
        assert(isinstance(text_parser, BaseTextParser))

        parsed_sentences = [text_parser.parse(news.sentence_to_terms_list(sent_ind))
                            for sent_ind in range(news.SentencesCount)]

        return ParsedNews(doc_id=news.ID,
                          parsed_sentences=parsed_sentences)
