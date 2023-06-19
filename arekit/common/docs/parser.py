from arekit.common.docs.base import Document
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.parser import BaseTextParser


class DocumentParser(object):

    @staticmethod
    def __get_sent(news, sent_ind):
        return news.get_sentence(sent_ind)

    @staticmethod
    def parse(news, text_parser, parent_ppl_ctx=None):
        assert(isinstance(news, Document))
        assert(isinstance(text_parser, BaseTextParser))
        assert(isinstance(parent_ppl_ctx, PipelineContext) or parent_ppl_ctx is None)

        parsed_sentences = [text_parser.run(input_data=DocumentParser.__get_sent(news, sent_ind).Text,
                                            params_dict=DocumentParser.__create_ppl_params(news=news, sent_ind=sent_ind),
                                            parent_ctx=parent_ppl_ctx)
                            for sent_ind in range(news.SentencesCount)]

        return ParsedDocument(doc_id=news.ID,
                              parsed_sentences=parsed_sentences)

    @staticmethod
    def __create_ppl_params(news, sent_ind):
        assert(isinstance(news, Document))
        return {
            "s_ind": sent_ind,                                  # sentence index. (as Metadata)
            "doc_id": news.ID,                                  # document index. (as Metadata)
            "sentence": DocumentParser.__get_sent(news, sent_ind),  # Required for special sources.
        }
