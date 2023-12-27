from arekit.common.docs.base import Document
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.parser import BaseTextParser


class DocumentParser(object):

    @staticmethod
    def __get_sent(doc, sent_ind):
        return doc.get_sentence(sent_ind)

    @staticmethod
    def parse(doc, text_parser, parent_ppl_ctx=None):
        assert(isinstance(doc, Document))
        assert(isinstance(text_parser, BaseTextParser))
        assert(isinstance(parent_ppl_ctx, PipelineContext) or parent_ppl_ctx is None)

        parsed_sentences = [text_parser.run(params_dict={"input": DocumentParser.__get_sent(doc, sent_ind)},
                                            parent_ctx=parent_ppl_ctx)
                            for sent_ind in range(doc.SentencesCount)]

        return ParsedDocument(doc_id=doc.ID,
                              parsed_sentences=parsed_sentences)
