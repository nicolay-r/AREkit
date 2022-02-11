from arekit.common.news.base import News
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.parser import BaseTextParser


class NewsParser(object):

    @staticmethod
    def parse(news, text_parser):
        assert(isinstance(news, News))
        assert(isinstance(text_parser, BaseTextParser))

        parsed_sentences = [text_parser.run(NewsParser.__create_pipeline_ctx(news, sent_ind))
                            for sent_ind in range(news.SentencesCount)]

        return ParsedNews(doc_id=news.ID,
                          parsed_sentences=parsed_sentences)

    @staticmethod
    def __create_pipeline_ctx(news, sent_ind):
        """ Default pipeline context.
        """
        assert(isinstance(news, News))

        sentence = news.get_sentence(sent_ind)

        return PipelineContext(d={
            "src": sentence.Text,                       # source data.
            "s_ind": sent_ind,                          # sentence index. (as Metadata)
            "doc_id": news.ID,                          # document index. (as Metadata)
            "sentence": sentence
        })
