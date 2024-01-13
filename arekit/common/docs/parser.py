from tqdm import tqdm
from arekit.common.docs.base import Document
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.batching import BatchingPipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.utils import BatchIterator
from arekit.common.text.parsed import BaseParsedText


class DocumentParsers(object):

    @staticmethod
    def parse(doc, pipeline_items, parent_ppl_ctx=None, src_key="input", show_progress=False):
        """ This document parser is based on single text parts (sentences)
            that passes sequentially through the pipeline of transformations.
        """
        assert(isinstance(doc, Document))
        assert(isinstance(pipeline_items, list))
        assert(isinstance(parent_ppl_ctx, PipelineContext) or parent_ppl_ctx is None)

        parsed_sentences = []

        data_it = range(doc.SentencesCount)
        progress_it = tqdm(data_it, disable=not show_progress)

        for sent_ind in progress_it:

            # Composing the context from a single sentence.
            ctx = PipelineContext({src_key: doc.get_sentence(sent_ind)}, parent_ctx=parent_ppl_ctx)

            # Apply all the operations.
            BasePipelineLauncher.run(pipeline=pipeline_items, pipeline_ctx=ctx, src_key=src_key)

            # Collecting the result.
            parsed_sentences.append(BaseParsedText(terms=ctx.provide("result")))

        return ParsedDocument(doc_id=doc.ID, parsed_sentences=parsed_sentences)

    @staticmethod
    def parse_batch(doc, pipeline_items, batch_size, parent_ppl_ctx=None, src_key="input", show_progress=False):
        """ This document parser is based on batch of sentences.
        """
        assert(isinstance(batch_size, int) and batch_size > 0)
        assert(isinstance(doc, Document))
        assert(isinstance(pipeline_items, list))
        assert(isinstance(parent_ppl_ctx, PipelineContext) or parent_ppl_ctx is None)

        parsed_sentences = []

        data_it = BatchIterator(lst=list(range(doc.SentencesCount)), batch_size=batch_size)
        progress_it = tqdm(data_it, total=round(doc.SentencesCount / batch_size), disable=not show_progress)

        for batch in progress_it:

            # Composing the context from a single sentence.
            ctx = PipelineContext({src_key: [doc.get_sentence(s_ind) for s_ind in batch]},
                                  parent_ctx=parent_ppl_ctx)

            # Apply all the operations.
            BatchingPipelineLauncher.run(pipeline=pipeline_items, pipeline_ctx=ctx, src_key=src_key)

            # Collecting the result.
            parsed_sentences += [BaseParsedText(terms=result) for result in ctx.provide("result")]

        return ParsedDocument(doc_id=doc.ID, parsed_sentences=parsed_sentences)
